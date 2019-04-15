---
layout:     post
title:      "dm_exe_request的应用"
subtitle:   "dm_exe_request tool"
date:       2017-04-02 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - SqlServer 
    - Database
---


## dm_exec_requests的使用

*Returns information about each request that is executing within SQL Server*  
查询当前SQL Server中正执行的请求  

Column Name|Data Type |Description|
---|---|---|
session_id 	|smallint 	|ID of the session to which this request is related. Is not nullable.|
request_id 	|int 	|ID of the request. Unique in the context of the session. Is not nullable.|

```sql
SELECT     
[Spid] = session_Id, ecid, [Database] = DB_NAME(sp.dbid),  
[User] = nt_username, [Status] = er.status, 
[Wait] = wait_type, 
[Individual Query] = SUBSTRING(qt.text, er.statement_start_offset / 2, 
                                   (CASE WHEN er.statement_end_offset = - 1 THEN LEN(CONVERT(NVARCHAR(MAX), qt.text))   * 2 
			     ELSE er.statement_end_offset END - er.statement_start_offset) / 2),
[Parent Query] = qt.text, 
Program = program_name, Hostname, 
nt_domain, 
start_time
FROM  sys.dm_exec_requests er INNER JOIN  sys.sysprocesses sp ON er.session_id = sp.spid 
CROSS APPLY sys.dm_exec_sql_text(er.sql_handle) AS qt
WHERE session_Id > 50 -- Ignore system spids.
AND session_Id NOT IN (@@SPID) 
and DB_NAME(sp.dbid)='CDR_ADT'  --dbname
```
