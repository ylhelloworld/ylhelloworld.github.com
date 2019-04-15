---
layout:     post
title:      "Job监测"
subtitle:   "Job监测及通知"
date:       2017-05-02 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post01.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - SqlServer
    - Database
---

SqlServer上有好多排程在运行，有时排遇到错误没有执行成功，或者被意外停止，此工具的作用就是监控排程的执行情况，并把监测情况短信通知给相关人员
### Refrences  

##### sysjobs  
存储将由 SQL Server 代理执行的各个预定作业的信息 
>  https://docs.microsoft.com/zh-cn/previous-versions/sql/sql-server-2005/ms189817(v%3dsql.90)
##### sysjobschedules 
存储将由 SQL Server 代理执行的各个预定作业的信息
> https://docs.microsoft.com/zh-cn/previous-versions/sql/sql-server-2005/ms188924(v%3dsql.90)

##### sysjobshistory  
包含有关 SQL Server 代理执行计划作业的信息。 此表存储在 msdb 数据库中。
> https://docs.microsoft.com/zh-cn/previous-versions/sql/sql-server-2005/ms174997(v%3dsql.90) 


### 监测工具
#### 基础表结构
```
CREATE TABLE [dbo].[JobRunLog](
	[Id] [INT] IDENTITY(1,1) NOT NULL,
	[RecordTime] [DATETIME] NULL,
	[Server] [NVARCHAR](500) NULL,
	[JobId] [UNIQUEIDENTIFIER] NULL,
	[JobName] [NVARCHAR](MAX) NULL,
	[LastRunTime] [DATETIME] NULL,
	[LastRunStatus] [INT] NULL,
	[LastRunStatusName] [NVARCHAR](50) NULL,
	[LastRunDuration] [NVARCHAR](50) NULL,
	[LastRunInfo] [NVARCHAR](MAX) NULL,
	[NextRunTime] [DATETIME] NULL,
 CONSTRAINT [PK_JobRunLog] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO

````


```
CREATE TABLE [dbo].[JobRunLogRemove](
	[id] [INT] IDENTITY(1,1) NOT NULL,
	[Server] [NVARCHAR](500) NULL,
	[JobId] [NVARCHAR](500) NULL,
	[JobName] [NVARCHAR](500) NULL,
	[JobDescription] [NVARCHAR](4000) NULL,
 CONSTRAINT [PK_Monitor_Job_Info] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

```

#### 监控脚本 
```
USE [CDR_Task]
GO

/****** Object:  StoredProcedure [dbo].[SP_JobRunStatuMonitor_APP]    Script Date: 2018/12/24 17:23:41 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO
ALTER PROCEDURE [dbo].[SP_JobRunStatuMonitor_APP] 
  
AS
BEGIN
 DECLARE @MaxRecordTime 	DATETIME
 DECLARE @Count1 INT 
 DECLARE @Count2 INT 
 DECLARE @Msg NVARCHAR(4000)
 DECLARE @Ip NVARCHAR(50)

 SET @Ip='192.168.204.42'
 SET @Msg=''
 SET @MaxRecordTime=ISNULL((SELECT MAX(RecordTime) FROM [dbo].[JobRunLog] WHERE [Server]=@Ip),'1900-01-01')
  INSERT INTO [dbo].[JobRunLog]
           (
		   RecordTime
		   ,[Server]
		   ,[JobId]
           ,[JobName]
           ,[LastRunTime]
           ,[LastRunStatus]
           ,[LastRunStatusName]
           ,[LastRunDuration]
           ,[LastRunInfo]
           ,[NextRunTime])
SELECT   * 
FROM 
(
SELECT  
		GETDATE() AS RecordTime,
		@Ip  AS [Server],
        [sJOB].[job_id] AS JobId ,
        [sJOB].[name] AS JobName ,
        CASE WHEN [sJOBH].[run_date] IS NULL
                  OR [sJOBH].[run_time] IS NULL THEN NULL
             ELSE CAST(CAST([sJOBH].[run_date] AS CHAR(8)) + ' '
                  + STUFF(STUFF(RIGHT('000000'
                                      + CAST([sJOBH].[run_time] AS VARCHAR(6)),
                                      6), 3, 0, ':'), 6, 0, ':') AS DATETIME)
        END AS LastRunTime ,
        [sJOBH].[run_status] LastRunStatus ,
        CASE [sJOBH].[run_status]
          WHEN 0 THEN '失败'
          WHEN 1 THEN '成功'
          WHEN 2 THEN '重试'
          WHEN 3 THEN '取消'
          WHEN 4 THEN '正在运行' -- In Progress
        END AS LastRunStatusName ,
        STUFF(STUFF(RIGHT('000000'
                          + CAST([sJOBH].[run_duration] AS VARCHAR(6)), 6), 3,
                    0, ':'), 6, 0, ':') AS [LastRunDuration (HH:MM:SS)] ,
        [sJOBH].[message] AS LastRunInfo ,
        CASE [sJOBSCH].[NextRunDate]
          WHEN 0 THEN NULL
          ELSE CAST(CAST([sJOBSCH].[NextRunDate] AS CHAR(8)) + ' '
               + STUFF(STUFF(RIGHT('000000'
                                   + CAST([sJOBSCH].[NextRunTime] AS VARCHAR(6)),
                                   6), 3, 0, ':'), 6, 0, ':') AS DATETIME)
        END AS NextRunTime 
FROM    [APPDB].[msdb].[dbo].[sysjobs] AS [sJOB]
        LEFT JOIN ( SELECT  [job_id] ,
                            MIN([next_run_date]) AS [NextRunDate] ,
                            MIN([next_run_time]) AS [NextRunTime]
                    FROM    [APPDB].[msdb].[dbo].[sysjobschedules]
                    GROUP BY [job_id]
                  ) AS [sJOBSCH] ON [sJOB].[job_id] = [sJOBSCH].[job_id]
        LEFT JOIN ( SELECT  [job_id] ,
                            [run_date] ,
                            [run_time] ,
                            [run_status] ,
                            [run_duration] ,
                            [message] ,
                            ROW_NUMBER() OVER ( PARTITION BY [job_id] ORDER BY [run_date] DESC, [run_time] DESC ) AS RowNumber
                    FROM    [msdb].[dbo].[sysjobhistory]
                    WHERE   [step_id] = 0
                  ) AS [sJOBH] ON [sJOB].[job_id] = [sJOBH].[job_id]
                                  AND [sJOBH].[RowNumber] = 1
) temp
WHERE temp.LastRunStatus=0 
AND (SELECT COUNT(1) FROM JobRunLog temp01 WHERE temp.JobId=temp01.JobId AND temp.LastRunTime=temp01.LastRunTime)=0
AND temp.JobId NOT IN (SELECT temp02.JobId FROM JObRunLogRemove temp02)

--无下次执行的时间的，一小时统计一次
 INSERT INTO [dbo].[JobRunLog]
           (
		   RecordTime
		   ,[Server]
		   ,[JobId]
           ,[JobName]
           ,[LastRunTime]
           ,[LastRunStatus]
           ,[LastRunStatusName]
           ,[LastRunDuration]
           ,[LastRunInfo]
           ,[NextRunTime])
SELECT   * 
FROM 
(
SELECT  
		GETDATE() AS RecordTime,
		@Ip AS [Server],
        [sJOB].[job_id] AS JobId ,
        [sJOB].[name] AS JobName ,
        CASE WHEN [sJOBH].[run_date] IS NULL
                  OR [sJOBH].[run_time] IS NULL THEN NULL
             ELSE CAST(CAST([sJOBH].[run_date] AS CHAR(8)) + ' '
                  + STUFF(STUFF(RIGHT('000000'
                                      + CAST([sJOBH].[run_time] AS VARCHAR(6)),
                                      6), 3, 0, ':'), 6, 0, ':') AS DATETIME)
        END AS LastRunTime ,
        [sJOBH].[run_status] LastRunStatus ,
        CASE [sJOBH].[run_status]
          WHEN 0 THEN '失败'
          WHEN 1 THEN '成功'
          WHEN 2 THEN '重试'
          WHEN 3 THEN '取消'
          WHEN 4 THEN '正在运行' -- In Progress
        END AS LastRunStatusName ,
        STUFF(STUFF(RIGHT('000000'
                          + CAST([sJOBH].[run_duration] AS VARCHAR(6)), 6), 3,
                    0, ':'), 6, 0, ':') AS [LastRunDuration (HH:MM:SS)] ,
        [sJOBH].[message] AS LastRunInfo ,
        CASE [sJOBSCH].[NextRunDate]
          WHEN 0 THEN NULL
          ELSE CAST(CAST([sJOBSCH].[NextRunDate] AS CHAR(8)) + ' '
               + STUFF(STUFF(RIGHT('000000'
                                   + CAST([sJOBSCH].[NextRunTime] AS VARCHAR(6)),
                                   6), 3, 0, ':'), 6, 0, ':') AS DATETIME)
        END AS NextRunTime 
FROM    [APPDB].[msdb].[dbo].[sysjobs] AS [sJOB]
        LEFT JOIN ( SELECT  [job_id] ,
                            MIN([next_run_date]) AS [NextRunDate] ,
                            MIN([next_run_time]) AS [NextRunTime]
                    FROM    [APPDB].[msdb].[dbo].[sysjobschedules]
                    GROUP BY [job_id]
                  ) AS [sJOBSCH] ON [sJOB].[job_id] = [sJOBSCH].[job_id]
        LEFT JOIN ( SELECT  [job_id] ,
                            [run_date] ,
                            [run_time] ,
                            [run_status] ,
                            [run_duration] ,
                            [message] ,
                            ROW_NUMBER() OVER ( PARTITION BY [job_id] ORDER BY [run_date] DESC, [run_time] DESC ) AS RowNumber
                    FROM    [APPDB].[msdb].[dbo].[sysjobhistory]
                    WHERE   [step_id] = 0
                  ) AS [sJOBH] ON [sJOB].[job_id] = [sJOBH].[job_id]
                                  AND [sJOBH].[RowNumber] = 1
) temp
WHERE temp.NextRunTime IS NULL 
--AND (SELECT COUNT(1) FROM JobRunLog temp01 WHERE temp.JobId=temp01.JobId AND DATEDIFF(HOUR,GETDATE(),temp.RecordTime)<1)=0
AND temp.JobId NOT IN (SELECT temp02.JobId FROM JObRunLogRemove temp02)
AND temp.JobId NOT IN  (
	SELECT  DISTINCT job_schedule.job_id
	FROM [APPDB].[msdb].[dbo].[sysjobschedules] AS  job_schedule
	LEFT JOIN [APPDB].[msdb].[dbo].[sysschedules] AS schedule ON job_schedule.[schedule_id] = schedule.[schedule_id]
	WHERE schedule.freq_type IN (1,64) OR schedule.enabled=0
) --随代理服务器启动，只执行一次，已禁用
ORDER BY temp.NextRunTime DESC

SET @Count1=(SELECT COUNT(1) FROM dbo.JobRunLog WHERE RecordTime>@MaxRecordTime AND LastRunStatus=0 AND [Server]=@Ip)
SET @Count2=(SELECT COUNT(1) FROM dbo.JobRunlog  WHERE RecordTime>@MaxRecordTime AND NextRunTime IS NULL AND [Server]=@Ip)
IF  @Count1>0   
BEGIN
	SET @Msg=@Ip+' JOB执行有报错，请及时查看日志'
END 

IF (@Count2>0)
BEGIN
	SET @Msg=@Ip+' JOB有被停止，请及时查看日志'
END 

IF (@Count1>0 AND @Count2>0)  
BEGIN 
	SET @Msg=@Ip+' JOB执行有报错 & JOB有被停止，请及时查看日志'
END 

PRINT(@Msg)
IF (@Msg<>'')
BEGIN
	PRINT(@Msg)
			INSERT INTO [192.168.204.41].[Uniform].[dbo].[IncomingMessage](
				 [Type]
				,[Header]
				,[Content]
				,[DeliveryMethod]
				,[SenderApp]
				,[SenderId]
				,[SenderName]
				,[SenderIP]
				,[Info]
				,[CreateTime]
				,[Status]
				,[ProcessServerName]
				,[ProcessServerIP]) 
				VALUES(
				 4,
				 'Job监控',
				 @Msg,
				 'Sms',
				 'Job监控',
				 'Job监控',
				 'CDR_Task.SP_JobRunStatuMonitor_SSO',
				 '192.168.204.29',
				 '[{"EmailAddress":"long.yang@eureka-systems.com","MobileNumber":"17301755856","UserId":"yanglong"}]' ,
				 GETDATE(),
				 0,
				 'CMSERVER',
				 '127.0.0.1'
				 )
  END 
END 

GO

```

