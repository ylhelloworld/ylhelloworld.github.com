---
layout:     post
title:      "dm_exe_query_status 的应用"
subtitle:   "dm_exe_query_status tool"
date:       2017-04-03 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - SqlServer 
    - Database
---



## dm_exec_query_stats的使用
*Returns aggregate performance statistics for cached query plans in SQL Server. The view contains one row per query statement within the cached plan, and the lifetime of the rows are tied to the plan itself. When a plan is removed from the cache, the corresponding rows are eliminated from this view.*  

查询返回SQLServer缓存的执行计划的性能统计，结果中每行包含每条查询计划的查询状态、有效期，档执行计划从缓存中移除时，对应的行记录也会从视图结果中移除

##### IO累计最高的查询语句
```sql
SELECT TOP 10
    [Average IO] = (total_logical_reads+total_logical_writes)/qs.execution_count
   ,[Total IO] = (total_logical_reads+total_logical_writes)
   ,[Execution count] = qs.execution_count
   ,[Individual Query] = SUBSTRING(qt.text,qs.statement_start_offset/2+1,
                                   (CASE WHEN qs.statement_end_offset=-1
                                         THEN LEN(CONVERT(NVARCHAR(MAX),qt.text))*2
                                         ELSE qs.statement_end_offset
                                    END-qs.statement_start_offset)/2)
   ,[Parent Query] = qt.text
   ,DatabaseName = DB_NAME(qt.dbid)
FROM
    sys.dm_exec_query_stats qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) AS qt
ORDER BY  [Total IO] DESC
```

##### 累计执行次数最多的查询语句
```sql
SELECT TOP 10
        total_worker_time
       ,plan_handle
       ,execution_count
       ,(SELECT SUBSTRING(text,statement_start_offset / 2 + 1,
                          (CASE WHEN statement_end_offset=-1
                                THEN LEN(CONVERT(NVARCHAR(MAX),text)) * 2
                                ELSE statement_end_offset
                           END - statement_start_offset) / 2)
         FROM   sys.dm_exec_sql_text(sql_handle)) AS query_text
FROM    sys.dm_exec_query_stats
ORDER BY execution_count DESC
```

##### 累计CPU时间最长的查询
```sql
SELECT TOP 10
        total_worker_time
       ,last_worker_time
       ,max_worker_time
       ,min_worker_time
       ,SUBSTRING(st.text,(qs.statement_start_offset / 2) + 1,
                  ((CASE statement_end_offset
                      WHEN -1 THEN DATALENGTH(st.text)
                      ELSE qs.statement_end_offset
                    END - qs.statement_start_offset) / 2) + 1) AS statement_text
FROM    sys.dm_exec_query_stats AS qs
CROSS   APPLY sys.dm_exec_sql_text(qs.sql_handle) AS st
ORDER BY total_worker_time DESC
```
##### 重编译最多的查询
```sql
SELECT TOP 10
        plan_generation_num
       ,execution_count
       ,(SELECT SUBSTRING(text,statement_start_offset / 2 + 1,
                          (CASE WHEN statement_end_offset=-1
                                THEN LEN(CONVERT(NVARCHAR(MAX),text)) * 2
                                ELSE statement_end_offset
                           END - statement_start_offset) / 2)
         FROM   sys.dm_exec_sql_text(sql_handle)) AS query_text
FROM    sys.dm_exec_query_stats
WHERE   plan_generation_num>1
ORDER BY plan_generation_num DESC
--过多的重编译会占用CPU资源，如果临时表中的数据量不多的话，可以考虑改成使用表变量，以减少重编译的次数
```

##### 综合查询
```sql 
SELECT  DB_ID(DB.dbid) '数据库名'
      , OBJECT_ID(db.objectid) '对象'
      , QS.creation_time '编译计划的时间'
      , QS.last_execution_time '上次执行计划的时间'
      , QS.execution_count '执行的次数'
      , QS.total_elapsed_time / 1000 '占用的总时间（秒）'
      , QS.total_physical_reads '物理读取总次数'
      , QS.total_worker_time / 1000 'CPU 时间总量（秒）'
      , QS.total_logical_writes '逻辑写入总次数'
      , QS.total_logical_reads N'逻辑读取总次数'
      , QS.total_elapsed_time / 1000 N'总花费时间（秒）'
      , SUBSTRING(ST.text, ( QS.statement_start_offset / 2 ) + 1,
                  ( ( CASE statement_end_offset
                        WHEN -1 THEN DATALENGTH(st.text)
                        ELSE QS.statement_end_offset
                      END - QS.statement_start_offset ) / 2 ) + 1) AS '执行语句'
FROM    sys.dm_exec_query_stats AS QS CROSS APPLY
        sys.dm_exec_sql_text(QS.sql_handle) AS ST INNER JOIN
        ( SELECT    *
          FROM      sys.dm_exec_cached_plans cp CROSS APPLY
                    sys.dm_exec_query_plan(cp.plan_handle)
        ) DB
            ON QS.plan_handle = DB.plan_handle
where   SUBSTRING(st.text, ( qs.statement_start_offset / 2 ) + 1,
                  ( ( CASE statement_end_offset
                        WHEN -1 THEN DATALENGTH(st.text)
                        ELSE qs.statement_end_offset
                      END - qs.statement_start_offset ) / 2 ) + 1) not like '%fetch%'
                      ORDER BY QS.total_elapsed_time / 1000 DESC 
```
