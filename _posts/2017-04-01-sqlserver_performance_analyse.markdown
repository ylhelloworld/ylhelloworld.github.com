---
layout:     post
title:      "SqlServer性能分析设计"
subtitle:   "SqlServer性能的记录&性能分析"
date:       2017-04-01 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - SqlServer 
    - Database
---



##  Performance  Tools  性能分析工具
数据性能分析主要使用的是SqlServer中的分析功能据dm_exec_request,dm_exec_query_statu以及当前数据库死锁状态三个工具进行分析
### dm_exec_request  
*Returns information about each request that is executing within SQL Server*  
查询当前SQL Server中正执行的请求  
```sql
SELECT
ID=NEWID(),
@MonitorID,
RequestTime=GETDATE(), 
SessionId=session_id,
RequestId=r.request_id,
[SQL]=t.text,
[StartTime]=r.start_time,
[Status]=r.status,
Command=r.command,
WaitType=r.wait_type,
WaitTime=r.wait_time,
CpuTime=r.cpu_time,
Reads=r.reads,
Writes=r.Writes, 
TotalElapsedTime=r.total_elapsed_time,
[RowCount]=r.row_count
FROM sys.dm_exec_requests r  
CROSS APPLY sys.dm_exec_sql_text(sql_handle) t 
WHERE r.start_time>=DATEADD(Hh,-24,GETDATE())

 ```

### dm_exec_query_stats   
*Returns aggregate performance statistics for cached query plans in SQL Server. The view contains one row per query statement within the cached plan, and the lifetime of the rows are tied to the plan itself. When a plan is removed from the cache, the corresponding rows are eliminated from this view.*  

查询返回SQLServer缓存的执行计划的性能统计，结果中包含每条查询计划的查询状态、有效期，档执行计划从缓存中移除时，对应的行记录也会从视图结果中移除
```sql
SELECT  
	   ID=NEWID()
	  ,MonitorID=@MonitorID
	  ,CreateTime=GETDATE()
	  ,CreationTime=QS.creation_time --编译计划的时间
	  , LastExecutionTime=QS.last_execution_time --'上次执行计划的时间'
	  , ExecutionCount=QS.execution_count --执行的次数
	  , TotalElapsedTime=QS.total_elapsed_time / 1000 --占用的总时间（毫秒）
	  , TotalPhysicalReads=QS.total_physical_reads --物理读取总次数
	  , TotalWorkedTime=QS.total_worker_time / 1000 --CPU 时间总量（毫秒）
	  , TotalLogicalWrites=QS.total_logical_writes --逻辑写入总次数
	  , TotalLogicalReads=QS.total_logical_reads --逻辑读取总次数 
	  , [SQL]=ST.text --执行语句
	  ,LastWorkerTime=QS.last_worker_time/ 1000
	  ,LastElapsedTime=QS.last_elapsed_time/ 1000
	  ,LastLogicalWrites=QS.last_logical_writes
	  ,LastLogicalReads=QS.last_logical_reads
	  ,LastPhysicalReads=qs.last_physical_reads
	  ,LastRows=QS.last_rows
FROM    sys.dm_exec_query_stats (NOLOCK) AS QS 
CROSS APPLY sys.dm_exec_sql_text(QS.sql_handle) AS ST 
WHERE QS.last_execution_time>DATEADD(Mi,-5,GETDATE())
ORDER BY QS.total_elapsed_time / 1000 DESC 
```

### sp_wholock

```sql
 SELECT 
  NEWID(),
@MonitorID,
GETDATE(),
TEMP_LOCK.SPID,
TEMP_LOCK.BLOCKED,
Remark= (CASE TEMP_LOCK.SPID
		WHEN 0 THEN '引起数据库死锁的是: '+ CAST(TEMP_LOCK.blocked AS VARCHAR(10)) + '进程号'
		ELSE  '进程号SPID：'+ CAST(spid AS VARCHAR(10))+ '被' + '进程号SPID：'+ CAST(TEMP_LOCK.blocked AS VARCHAR(10)) +'阻塞'
		END),
SQLContent= qt.text
FROM 
(
	SELECT  0 AS SPID ,blocked ,a.sql_handle
	FROM (SELECT * FROM sysprocesses WHERE  blocked>0 ) a 
	WHERE NOT EXISTS(SELECT * FROM (SELECT * FROM sysprocesses WHERE  blocked>0 ) b  WHERE a.blocked=spid) 
	UNION 
	SELECT spid,blocked ,sql_handle FROM sysprocesses WHERE  blocked>0 
) TEMP_LOCK
CROSS APPLY sys.dm_exec_sql_text(TEMP_LOCK.sql_handle) AS qt

```

## Performance Design
每隔固定的时间点，去记录当前数据库的请求状态，执行计划，和死锁状态，从这3个方面分析数据库的性能和数据库的使用情况。
- 由于存储限制，每半小时记录一次数据库状态*[后续状态指的就是当前请求状态、执行计划、死锁状态]*  ，只保留3个月的数据，根据存储大小和具体分析需要可在脚本中进行调整
- 需要准备一组本数据库经常查询的存储过程，用于记录该过程的每次执行时间，作为数据库状态性能的基准   

### Create Table  
```sql
CREATE TABLE [dbo].[Log_Monitor](
	[id] [UNIQUEIDENTIFIER] NOT NULL,
	[name] [NVARCHAR](4000) NULL,
	[CreateTime] [DATETIME] NULL,
	[StartTime] [DATETIME] NULL,
	[EndTime] [DATETIME] NULL,
	[UsedTime] [INT] NULL,
	[Remark] [NVARCHAR](4000) NULL,
 CONSTRAINT [PK_Log_Monitor] PRIMARY KEY NONCLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]


CREATE TABLE [dbo].[Log_Monitor_Locked](
	[id] [UNIQUEIDENTIFIER] NOT NULL,
	[MonitorID] [UNIQUEIDENTIFIER] NULL,
	[CreateTime] [DATETIME] NULL,
	[SPID] [NVARCHAR](50) NULL,
	[BLOCKED] [NVARCHAR](50) NULL,
	[Remark] [NVARCHAR](4000) NULL,
	[SQL] [NTEXT] NULL,
 CONSTRAINT [PK_Log_Monitor_Locked] PRIMARY KEY NONCLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]



CREATE TABLE [dbo].[Log_Monitor_QueryState](
	[id] [UNIQUEIDENTIFIER] NOT NULL,
	[MonitorID] [UNIQUEIDENTIFIER] NULL,
	[CreateTime] [DATETIME] NULL,
	[CreationTime] [DATETIME] NULL,
	[ExecutionCount] [BIGINT] NULL,
	[SQL] [NTEXT] NULL,
	[LastExecutionTime] [DATETIME] NULL,
	[LastElapsedTime] [BIGINT] NULL,
	[LastPhysicalReads] [BIGINT] NULL,
	[LastLogicalReads] [BIGINT] NULL,
	[LastLogicalWrites] [BIGINT] NULL,
	[LastWorkerTime] [BIGINT] NULL,
	[LastRows] [BIGINT] NULL,
	[TotalElapsedTime] [BIGINT] NULL,
	[TotalWorkedTime] [BIGINT] NULL,
	[TotalLogicalReads] [BIGINT] NULL,
	[TotalLogicalWrites] [BIGINT] NULL,
	[TotalPhysicalReads] [BIGINT] NULL,
 CONSTRAINT [PK_Log_Monitor_QueryState] PRIMARY KEY NONCLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]


CREATE TABLE [dbo].[Log_Monitor_Request](
	[ID] [UNIQUEIDENTIFIER] NOT NULL,
	[MonitorID] [UNIQUEIDENTIFIER] NULL,
	[CreateTime] [DATETIME] NULL,
	[SessionId] [NVARCHAR](50) NULL,
	[RequestId] [NVARCHAR](50) NULL,
	[SQL] [NTEXT] NULL,
	[StartTime] [DATETIME] NULL,
	[Status] [NVARCHAR](50) NULL,
	[Command] [NVARCHAR](50) NULL,
	[WaitType] [NVARCHAR](50) NULL,
	[WaitTime] [INT] NULL,
	[CpuTime] [INT] NULL,
	[Reads] [BIGINT] NULL,
	[Writes] [BIGINT] NULL,
	[TotalElapsedTime] [INT] NULL,
	[RowCount] [BIGINT] NULL,
 CONSTRAINT [PK_Log_Monitor_Request] PRIMARY KEY NONCLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

```
#### Create Index     
创建索引的目的，主要为后续的性能分析可能因为数据量的增多变慢，从而提高查询速度  
```sql  
CREATE CLUSTERED INDEX [Index_CreateTime] ON [dbo].[Log_Monitor]
(
	[CreateTime] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
CREATE CLUSTERED INDEX [Index_CreateTime] ON [dbo].[Log_Monitor_Locked]
(
	[CreateTime] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
CREATE CLUSTERED INDEX [Index_CreateTime] ON [dbo].[Log_Monitor_QueryState]
(
	[CreateTime] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
CREATE CLUSTERED INDEX [Index_CreateTime] ON [dbo].[Log_Monitor_Request]
(
	[CreateTime] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO

```

### SP_PerformanceMonitor
创建存储过程 SP_PersormanceMonitor ,这个存储过程主要目的为记录当前数据库状态  
```sql
Create  PROCEDURE [dbo].[SP_PerformanceMonitor]
AS 
 DECLARE @CreateTime DATETIME
 DECLARE @StartTime DATETIME
 DECLARE @EndTime DATETIME
 DECLARE @MonitorID UNIQUEIDENTIFIER
    BEGIN
	   SET @CreateTime=GETDATE()
	   SET @MonitorID=NEWID()
	   PRINT(@StartTime)
	   --死锁情况
	  -- DELETE FROM  [dbo].Log_Monitor_Locked WHERE CreateTime < DATEADD(Mm,-1,GETDATE());
	   INSERT INTO Log_Monitor_Locked
	   (
	     id ,
		 MonitorID,
		 CreateTime,
	     SPID,
		 BLOCKED,
		 Remark,
		 [SQL]
	   )
	   SELECT 
	    NEWID(),
		@MonitorID,
		GETDATE(),
		TEMP_LOCK.SPID,
		TEMP_LOCK.BLOCKED,
		Remark= (CASE TEMP_LOCK.SPID
				WHEN 0 THEN '引起数据库死锁的是: '+ CAST(TEMP_LOCK.blocked AS VARCHAR(10)) + '进程号'
				ELSE  '进程号SPID：'+ CAST(spid AS VARCHAR(10))+ '被' + '进程号SPID：'+ CAST(TEMP_LOCK.blocked AS VARCHAR(10)) +'阻塞'
				END),
		SQLContent= qt.text
		FROM 
		(
			SELECT  0 AS SPID ,blocked ,a.sql_handle
			FROM (SELECT * FROM sysprocesses WHERE  blocked>0 ) a 
			WHERE NOT EXISTS(SELECT * FROM (SELECT * FROM sysprocesses WHERE  blocked>0 ) b  WHERE a.blocked=spid) 
			UNION 
			SELECT spid,blocked ,sql_handle FROM sysprocesses WHERE  blocked>0 
		) TEMP_LOCK
		CROSS APPLY sys.dm_exec_sql_text(TEMP_LOCK.sql_handle) AS qt

		--记录当前请求状态
		--DELETE FROM  [dbo].[Log_Monitor_Request] WHERE CreateTime < DATEADD(Mm,-1,GETDATE());
		INSERT INTO [dbo].[Log_Monitor_Request]
           ([ID]
           ,[MonitorID]
           ,[CreateTime]
           ,[SessionId]
           ,[RequestId]
           ,[SQL]
           ,[StartTime]
           ,[Status]
           ,[Command]
           ,[WaitType]
           ,[WaitTime]
		   ,CpuTime
		   ,Reads
		   ,Writes
		   ,TotalElapsedTime
		   ,[RowCount]
		   ) 
		  SELECT
			ID=NEWID(),
			@MonitorID,
			RequestTime=GETDATE(), 
			SessionId=session_id,
			RequestId=r.request_id,
			[SQL]=t.text,
			[StartTime]=r.start_time,
			[Status]=r.status,
			Command=r.command,
			WaitType=r.wait_type,
			WaitTime=r.wait_time,
			CpuTime=r.cpu_time,
			Reads=r.reads,
			Writes=r.Writes, 
			TotalElapsedTime=r.total_elapsed_time,
			[RowCount]=r.row_count
			FROM sys.dm_exec_requests r  
			CROSS APPLY sys.dm_exec_sql_text(sql_handle) t  
			WHERE r.start_time>=DATEADD(Hh,-24,GETDATE())
	   --执行统计
	   --DELETE FROM  [dbo].[Log_Monitor_QueryState] WHERE CreateTime < DATEADD(Hh,-24,GETDATE());
	   INSERT INTO [dbo].[Log_Monitor_QueryState]
           ([id]
           ,[MonitorID]
           ,[CreateTime]
           ,[CreationTime]
           ,[LastExecutionTime]
           ,[ExecutionCount]
           ,[TotalElapsedTime]
           ,[TotalPhysicalReads]
           ,[TotalWorkedTime]
           ,[TotalLogicalWrites]
           ,[TotalLogicalReads]
           ,[SQL]
		   ,LastWorkerTime
		   ,LastElapsedTime
		   ,LastLogicalWrites
		   ,LastLogicalReads
		   ,LastPhysicalReads
		   ,LastRows
		   )
			SELECT  
				   ID=NEWID()
				  ,MonitorID=@MonitorID
				  ,CreateTime=GETDATE()
				  ,CreationTime=QS.creation_time --编译计划的时间
				  , LastExecutionTime=QS.last_execution_time --'上次执行计划的时间'
				  , ExecutionCount=QS.execution_count --执行的次数
				  , TotalElapsedTime=QS.total_elapsed_time / 1000 --占用的总时间（毫秒）
				  , TotalPhysicalReads=QS.total_physical_reads --物理读取总次数
				  , TotalWorkedTime=QS.total_worker_time / 1000 --CPU 时间总量（毫秒）
				  , TotalLogicalWrites=QS.total_logical_writes --逻辑写入总次数
				  , TotalLogicalReads=QS.total_logical_reads --逻辑读取总次数 
				  , [SQL]=ST.text --执行语句
				  ,LastWorkerTime=QS.last_worker_time/ 1000
				  ,LastElapsedTime=QS.last_elapsed_time/ 1000
				  ,LastLogicalWrites=QS.last_logical_writes
				  ,LastLogicalReads=QS.last_logical_reads
				  ,LastPhysicalReads=qs.last_physical_reads
				  ,LastRows=QS.last_rows
			FROM    sys.dm_exec_query_stats (NOLOCK) AS QS 
			CROSS APPLY sys.dm_exec_sql_text(QS.sql_handle) AS ST 
			WHERE QS.last_execution_time>DATEADD(Mi,-5,GETDATE())
			ORDER BY QS.total_elapsed_time / 1000 DESC 


	   SET @StartTime=GETDATE()
	   --------------------------------------
		  
		 EXEC [SurveillanceSystem].[dbo].[SP_MonitorSql]

	   -------------------------------------- 
	   SET @EndTime=GETDATE()

   
       --记录执行时间
	   --DELETE FROM  [dbo].[Log_Monitor] WHERE CreateTime < DATEADD(Mm,-12,GETDATE());
	   INSERT INTO [dbo].[Log_Monitor]
           ([id]
           ,[name]
		   ,[CreateTime]
           ,[StartTime]
           ,[EndTime]
           ,[UsedTime]
           ,[Remark])
		 VALUES
			   (@MonitorID
			   ,'院感执行速度监控'
			   ,@CreateTime
			   ,@StartTime
			   ,@EndTime
			   ,DATEDIFF( MILLISECOND, @StartTime, @EndTime )
			   ,'')
     
    END

```

###  Create Job
创建Job 用于设置记录存储过程的频次以及数据清除的频次   
```sql
Create  Job  Job_PerformanceMonitor
Create  Job  Job_PerformanceMonitor_Clear
```

### Performance Analyse

下面这个列子为分析近两月的每个小时节点的数据库性能，可以明显发现每两个小时，数据库性能有明细的下降，根据记录日志再分析是数据库定时转储引起  

2018年三月份的每个小时的性能分析
![IMAGE](http://ylhelloworld.github.io/img/resource/20180628001_chart.png)


2018年二月份~三月份的每个小时的性能分析
![IMAGE](http://ylhelloworld.github.io/img/resource/20180628002_chart.png)

