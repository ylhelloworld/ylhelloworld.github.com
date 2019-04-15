---
layout:     post
title:      "IIS日志分析"
subtitle:   "IIS日志存储&日志分析"
date:       2017-05-01 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - C#.Net Framerwork
--- 

## Why use Log Parser? 
Log parser is a powerful, versatile tool that provides universal query access to text-based data such as log files, XML files and CSV files, as well as key data sources on the Windows® operating system such as the Event Log, the Registry, the file system, and Active Directory®. You tell Log Parser what information you need and how you want it processed. The results of your query can be custom-formatted in text based output, or they can be persisted to more specialty targets like SQL, SYSLOG, or a chart. 

Most software is designed to accomplish a limited number of specific tasks. Log Parser is different... the number of ways it can be used is limited only by the needs and imagination of the user. The world is your database with Log Parser.

## How use Log Parser? 

#### 日志位置
日志对应关系存放位置：C:\windows\system32\inetsrv\config\applicationHost.config

#### 工具导入
- 安装Log Parse 2.2   [[下载地址]](http://www.microsoft.com/en-us/download/details.aspx?id=24659)
- 执行命令，切换目录 
```code 
> CD  C:\Program Files (x86)\Log Parser 2.2  
```
- 执行命令,插入数据库

**本机测试**
```code
LogParser   "SELECT  *  FROM  'C:\test\log\u_ex171031.log'  to xxxx_IISLOG"  -i:IISW3C  -o:SQL  -oConnString:"Driver={SQL Server};server=.;database=Data;IntegratedSecurity=SSPI"  -createtable:ON
LogParser   "SELECT  *  FROM  'C:\test\log\u_ex171030.log'  to xxxx_IISLOG"  -i:IISW3C  -o:SQL  -oConnString:"Driver={SQL Server};server=.;database=Data;IntegratedSecurity=SSPI"  -createtable:ON
```
 
#### 日志分析
 
```sql
--楼层
SELECT  * FROM   [xxxx_Floor] 
--登录日志
SELECT  * FROM   [xxxx_User_LoginLog]
--统计访问机器、IP、使用人员
SELECT  Logfilename,date,time,cip,sIP, sport,l.csMethod,l.csUriStem,csuseragent,f.HospitalName,f.FloorCode,f.FloorName,f.AreaCode,f.AreaName,f.RoomCode,f.RoomName,f.MachineName
FROM [CDR_API].[dbo].[xxxx_IISLOG] l
ORDER BY  time DESC
--统计点击量
SELECT  DISTINCT logfilename,[time],l.csUriStem
FROM [CDR_API].[dbo].[xxxx_IISLOG] l
WHERE l.csUriStem LIKE '/Patient/VisitNumber%'
OR l.csUriStem  LIKE '/EMR/Display/%'
ORDER BY l.time

``` 

#### 引用资料
> [1]   https://technet.microsoft.com/en-us/scriptcenter/dd919274.aspx  

  