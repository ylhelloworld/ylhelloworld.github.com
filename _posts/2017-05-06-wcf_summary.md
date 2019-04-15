---
layout:     post
title:      " WCF的应用"
subtitle:   "WCF的简介&应用"
date:       2017-05-06 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - C#.Net Framework
---
### Why use WCF?
Windows Communication Foundation(WCF)是由微软发展的一组数据通信的应用程序开发接口，可以翻译为Windows通讯接口，它是.NET框架的一部分。由 .NET Framework 3.0 开始引入。
- 整合了.Net平台下所有的和分布式系统有关的技术，如Enterprise Sevices(COM+)、.Net Remoting、Web Service(ASMX)、WSE3.0和MSMQ消息队列,
- 可以以ASP.NET，EXE，WPF，Windows Forms，NT Service，COM+作为宿主(Host)
- WCF可以支持的协议包括TCP，HTTP，跨进程以及自定义，安全模式则包括SAML， Kerberos，X509，用户/密码，自定义等多种标准与模式
##### Refrence 
>  https://msdn.microsoft.com/en-us/library/dd456779(v=vs.110).aspx 

  
### Binding


#### Binding描述信息  

| 层次 | 备注说明 | 
|:--------|:-------|
| Transactions（事务） | TransactionFlowBindingElement，用于指定事务流程 |
| Reliability（信赖） | ReliableSessionBindingElement，用于指定对会话方式|
| Security（安全） | SecurityBindingElement，指定安全方式|
| Encoding（编码） | Text, Binary, MTOM, Custom，指定数据传输格式|
| Transport（传输） | TCP, Named Pipes, HTTP, HTTPS, MSMQ, Custom，指定传输方式|


#### Binding 种类
- asicHttpBinding: 最简单的绑定类型，通常用于 Web Services。使用 HTTP 协议，Text/XML 编码方式。
- WSHttpBinding: 比 BasicHttpBinding 更加安全，通常用于 non-duplex 服务通讯。
- WSDualHttpBinding: 和 WSHttpBinding 相比，它支持 duplex 类型的服务。
- WSFederationHttpBinding: 支持 WS-Federation 安全通讯协议。
- NetTcpBinding: 效率最高，安全的跨机器通讯方式。
- NetNamedPipeBinding: 安全、可靠、高效的单机服务通讯方式。
- NetMsmqBinding: 使用消息队列在不同机器间进行通讯。
- NetPeerTcpBinding: 使用 P2P 协议在多机器间通讯。
- MsmqIntegrationBinding: 使用现有的消息队列系统进行跨机器通讯。如 MSMQ。 

|Binding | Configuration Element | Description| 
| ---|---|---| 
| BasicHttpBinding | basicHttpBinding | 一个指定用符合基本网络服务规范通讯的binding，它用http进行传输，数据格式为text/xml| 
| WSHttpBinding | wsHttpBinding | 一个安全的通用的binding,但它不能在deplex中使用|
| WSDualHttpBinding | wsDualHttpBinding | 一个安全的通用的binding,但能在deplex中使用| 
| WSFederationHttpBinding | wsFederationHttpBinding | 一个安全的通用的支持WSF的binding，能对用户进行验证和授权| 
| NetTcpBinding | netTcpBinding | 在wcf应用程序中最适合跨机器进行安全通讯的binding| 
| NetNamedPipeBinding | netNamedPipeBinding | 在wcf应用程序中最适合本机进行安全通讯的binding| 
| NetMsmqBinding | netMsmqBinding | 在wcf应用程序中最适合跨机器进行安全通讯的binding，并且支持排队| 
| NetPeerTcpBinding | netPeerTcpBinding | 一个支持安全的，多机交互的binding| 
| MsmqIntegrationBinding | msmqIntegrationBinding | 一个用于wcf与现有msmq程序进行安全通讯的binding|


##### Bingding比较
 互操作性，安全性，支持回话，支持事务，是否为全双工

Bingding | Interoperability | Security | Session | Transactions | Duplex
--- | --- | --- | --- | --- | ---
BasicHttpBinding | 	Basic Profile 1.1 | 	(None), Transport, Message | 	None, (None) | 	None | 	n/a
WSHttpBinding	 | WS | 	Transport, (Message), Mixed	 | (None), Transport, Reliable Session | 	(None), Yes	 | n/a
WSDualHttpBinding | 	WS | 	(Message) | 	(Reliable Session) | 	(None), Yes | 	Yes
WSFederationHttpBinding | 	WS-Federation | 	(Message)	 | (None), Reliable Session	 | (None), Yes | 	No
NetTcpBinding | 	.NET | 	(Transport), Message | 	Reliable Session, (Transport) | 	(None), Yes	 | Yes
NetNamedPipeBinding | 	.NET | 	(Transport)	 | None, (Transport)	 | (None), Yes | 	Yes
NetMsmqBinding | 	.NET | 	Message, (Transport), Both | 	(None) | 	(None), Yes | 	No
NetPeerTcpBinding | 	Peer | 	(Transport) | 	(None) | 	(None) | 	Yes
MsmqIntegrationBinding | 	MSMQ | 	(Transport) | 	(None) | 	(None), Yes	 | n/a

### How use wcf?
 
#### 服务器端 

##### 服务器端配置说明
```xml
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
    <!-- <system.ServiceModel> section -->
    <system.ServiceModel>
        <!-- services 元素包含应用中驻留的所有service的配置要求 -->
        <services>
            <!-- 
                 每个服务的配置 
                 属性说明: 
             name - 指定这个service配置是针对的那个服务,为一个实现了某些Contract的服务类的完全限定名 (名称空间.类型名),ServiceHost载入一个服务后，会到配置文件中<services>下找有没有name属性跟服务匹配的<service>的配置 
             behaviorConfiguration - 指定在<serviceBehaviors>下的一个<behavior>的name,这个特定<behavior>给这个service制定了一些行为,比如服务是否允许身份模拟-->
            <service name="名称空间.类型名" behaviorConfiguration="behavior名">
                <!-- 每个服务可以有多个Endpoint，下面<endpoint>元素对每个Endpoint分别进行配置
                 属性说明: 
                 address -服务端地址， 指定这个Endpoint对外的URI,这个URI可以是个绝对地址，也可以是个相对于baseAddress的相对地址。如果此属性为空，则这个Endpoint的地址就是baseAddress
                 binding -绑定协议， 指定这个Endpoint使用的binding，这个banding可以是系统预定义的9个binding之一， 比如是basicHttpBinding，也可以是自定义的customBinding。binding决定了通讯的类型、安全、如何编码、是否基于session、是否基于事务等等
                 contract - 协议全限定名，指定这个Endpoint对应的Contract的全限定名(名称空间.类型名)，这个Contract应该被  service元素的name指定的那个service实现 
                 bindingConfiguration - 指定一个binding的配置名称，跟<bindings>下面同类<binding>的name匹配
                 name - Endpoint的名称，可选属性，每个Contract都可以有多个Endpoint，但是每个Contract对应的多个Endpoint名必须是唯一的-->
                <endpoint address="URI" binding="basicHttpBinding"  bindingConfiguration="binding名" contract="Contract全限定名"  name="">
                    <!-- 用户定义的xml元素集合，一般用作SOAP的header内容-->
                    <headers>
                        <!-- 任何xml内容 -->
                    </headers>
                    <identity>
                        <!-- <identity>下的元素都是可选的-->
                        <userPrincipalName></userPrincipalName>
                        <servicePrincipalName></servicePrincipalName>
                        <dns></dns>
                        <rsa></rsa>
                        <certificate encodedValue=""></certificate>
                        <!-- <certificateReference>的属性都是可选的
                         属性说明：
                         storeName - 证书的存储区，可能值为：AddressBook，AuthRoot，CertificateAuthority Disallowed，My，Root，TrustedPeople，TrustedPublisher
                         storeLocation - 证书存储位置，可能值为：CurrentUser，LocalMachine-->
                        <certificateReference storeName="" storeLocation="">
                        </certificateReference>
                    </identity>
                </endpoint>
                <host>
                    <baseAddresses>
                        <!-- 在此可以定义每种传输协议的baseAddress，用于跟使用同样传输协议Endpoint定义的相对地
                    址组成完整的地址，但是每种传输协议只能定义一个baseAddress。HTTP的baseAddress同时是service
                    对外发布元数据的URL-->
                        <add baseAddress="http://address" />
                    </baseAddresses>
                    <timeouts></timeouts>
                </host>
            </service>
        </services>

        <bindings>
            <!-- 指定一个或多个系统预定义的binding，比如<basicHttpBinding>，当然也可以指定自定义的customBinding，
             然后在某个指定的binding下建立一个或多个配置，以便被Endpoint来使用这些配置 -->
            <basicHttpBinding>
                <!-- 某一类的binding的下面可能有多个配置，binding元素的name属性标识某个binding-->
                <binding name="binding名">
                </binding>
            </basicHttpBinding>
        </bindings>
        <!-- 定义service和Endpiont行为-->
        <behaviors>
            <!-- 定义service的行为-->
            <serviceBehaviors>
                <!-- 一个或多个系统提供的或定制的behavior元素
                 属性说明：
                 name - 一个behavior唯一标识,<service>元素的behaviorConfiguration属性指向这个name-->
                <behavior name="">
                    <!-- 指定service元数据发布和相关信息
                     属性说明：
                     httpGetEnabled - bool类型的值，表示是否允许通过HTTP的get方法获取sevice的WSDL元数据
                     httpGetUrl - 如果httpGetEnabled为true，这个属性指示使用哪个URL地址发布服务的WSDL，  如果这个属性没有设置，则使用服务的HTTP类型的baseAddress后面加上?WSDL-->
                    <serviceMetadata httpGetEnabled="true" httpGetUrl="http://URI:port/address" />
                </behavior>
            </serviceBehaviors>
            <!-- 定义Endpiont的行为-->
            <endpointBehaviors>
            </endpointBehaviors>
        </behaviors>
        <!-- 包含客户端跟服务端连接使用到的Endpoint的配置 -->
        <client>
            <!-- 每个客户端Endpoint设置
             属性说明：
             address - 对应到服务端这个Endpoint的address
             binding - 指定这个Endpoint使用的binding，这个banding可以是系统预定义的9个binding之一， 比如是basicHttpBinding
             contract - 指定这个Endpoint对应的Contract的全限定名(名称空间.类型名)
             name - Endpoint的配置名，客户端代理类的构造方法中的endpointConfigurationName对应到这个name
             bindingConfiguration - 指定客户端binding的具体设置，指向<bindings>元素下同类型binding的name -->
            <endpoint address="URI"
                binding="basicHttpBinding" bindingConfiguration="binding名"
                contract="Contract全限定名" name="endpoint配置名" />
        </client>
    </system.ServiceModel>
</configuration> 

```



#####  定义接口
```cs
// IService.cs  
using System;  
using System.Collections.Generic;  
using System.Linq;  
using System.Runtime.Serialization;  
using System.ServiceModel;  
using System.Text;  

namespace GettingStartedLib  
{  
        [ServiceContract(Namespace = "http://Microsoft.ServiceModel.Samples")]  
        public interface ICalculator  
        {  
            [OperationContract]  
            double Add(double n1, double n2);  
            [OperationContract]  
            double Subtract(double n1, double n2);  
            [OperationContract]  
            double Multiply(double n1, double n2);  
            [OperationContract]  
            double Divide(double n1, double n2);  
        }  
}  
```
##### 实现接口 
```cs
//Service1.cs  
using System;  
using System.Collections.Generic;  
using System.Linq;  
using System.Runtime.Serialization;  
using System.ServiceModel;  
using System.Text;  

namespace GettingStartedLib  
{  
    public class CalculatorService : ICalculator  
    {  
        public double Add(double n1, double n2)  
        {  
            double result = n1 + n2;  
            Console.WriteLine("Received Add({0},{1})", n1, n2);  
            // Code added to write output to the console window.  
            Console.WriteLine("Return: {0}", result);  
            return result;  
        }  

        public double Subtract(double n1, double n2)  
        {  
            double result = n1 - n2;  
            Console.WriteLine("Received Subtract({0},{1})", n1, n2);  
            Console.WriteLine("Return: {0}", result);  
            return result;  
        }  

        public double Multiply(double n1, double n2)  
        {  
            double result = n1 * n2;  
            Console.WriteLine("Received Multiply({0},{1})", n1, n2);  
            Console.WriteLine("Return: {0}", result);  
            return result;  
        }  

        public double Divide(double n1, double n2)  
        {  
            double result = n1 / n2;  
            Console.WriteLine("Received Divide({0},{1})", n1, n2);  
            Console.WriteLine("Return: {0}", result);  
            return result;  
        }  
    }  
}
```
- [ServiceContract]，来说明接口是一个WCF的接口，如果不加的话，将不能被外部调用。
- [OperationContract]，来说明该方法是一个WCF接口的方法，不加的话同上。
 
##### 运行Service
- 以IIS作为宿主运行
- 以Console Service 作为宿主运行


```cs
using System;  
using System.Collections.Generic;  
using System.Linq;  
using System.Text;  
using System.ServiceModel;  
using GettingStartedLib;  
using System.ServiceModel.Description;   

namespace GettingStartedHost  
{  
    class Program  
    {  
        static void Main(string[] args)  
        {  
            // Step 1 Create a URI to serve as the base address.  
            Uri baseAddress = new Uri("http://localhost:8000/GettingStarted/");  

            // Step 2 Create a ServiceHost instance  
            ServiceHost selfHost = new ServiceHost(typeof(CalculatorService), baseAddress);  

            try  
            {  
                // Step 3 Add a service endpoint.  
                selfHost.AddServiceEndpoint(typeof(ICalculator), new WSHttpBinding(), "CalculatorService");  

                // Step 4 Enable metadata exchange.  
                ServiceMetadataBehavior smb = new ServiceMetadataBehavior();  
                smb.HttpGetEnabled = true;  
                selfHost.Description.Behaviors.Add(smb);  

                // Step 5 Start the service.  
                selfHost.Open();  
                Console.WriteLine("The service is ready.");  
                Console.WriteLine("Press <ENTER> to terminate service.");  
                Console.WriteLine();  
                Console.ReadLine();  

                // Close the ServiceHostBase to shutdown the service.  
                selfHost.Close();  
            }  
            catch (CommunicationException ce)  
            {  
                Console.WriteLine("An exception occurred: {0}", ce.Message);  
                selfHost.Abort();  
            }  
        }  
    }  
}  
```

#### 客户端
##### 创建客户端
- Create a new console application project by right-clicking on the Getting Started solution, selecting, Add, New Project. In the Add New Project dialog on the left hand side of the dialog select Windows under C# or VB. In the center section of the dialog select Console Application. Name the project GettingStartedClient.

- Set the target framework of the GettingStartedClient project to .NET Framework 4.5 by right clicking on GettingStartedClient in the Solution Explorer and selecting Properties. In the dropdown box labeled Target Framework select .NET Framework 4.5. Setting the target framework for a VB project is a little different, in the GettingStartedClient project properties dialog, click the Compile tab on the left-hand side of the screen, and then click the Advanced Compile Options button at the lower left-hand corner of the dialog. Then select .NET Framework 4.5 in the dropdown box labeled Target Framework.Setting the target framework will cause Visual Studio 2011 to reload the solution, press OK when prompted.

- Add a reference to System.ServiceModel to the GettingStartedClient project by right-clicking the Reference folder under the GettingStartedClient project in Solution Explorer and select Add Reference. In the Add Reference dialog select Framework on the left-hand side of the dialog. In the Search Assemblies textbox, type in System.ServiceModel. In the center section of the dialog select System.ServiceModel, click the Add button, and click the Close button. Save the solution by clicking the Save All button below the main menu.

- Next you wlll add a service reference to the Calculator Service. Before you can do that, you must start up the GettingStartedHost console application. Once the host is running you can right click the References folder under the GettingStartedClient project in the Solution Explorer and select Add Service Reference and type in the following URL in the address box of the Add Service Reference dialog: HYPERLINK "http://localhost:8000/ServiceModelSamples/Service" http://localhost:8000/ServiceModelSamples/Service and click the Go button. The CalculatorService should then be displayed in the Services list box, Double click CalculatorService and it will expand and show the service contracts implemented by the service. Leave the default namespace as is and click the OK button.

##### 配置客户端  
```xml
<?xml version="1.0" encoding="utf-8" ?>  
<configuration>  
    <startup>   
      <!-- specifies the version of WCF to use-->  
        <supportedRuntime version="v4.0" sku=".NETFramework,Version=v4.5,Profile=Client" />  
    </startup>  
    <system.serviceModel>  
        <bindings>  
            <!-- Uses wsHttpBinding-->  
            <wsHttpBinding>  
                <binding name="WSHttpBinding_ICalculator" />  
            </wsHttpBinding>  
        </bindings>  
        <client>  
            <!-- specifies the endpoint to use when calling the service -->  
            <endpoint address="http://localhost:8000/ServiceModelSamples/Service/CalculatorService"  
                binding="wsHttpBinding" bindingConfiguration="WSHttpBinding_ICalculator"  
                contract="ServiceReference1.ICalculator" name="WSHttpBinding_ICalculator">  
                <identity>  
                    <userPrincipalName value="migree@redmond.corp.microsoft.com" />  
                </identity>  
            </endpoint>  
        </client>  
    </system.serviceModel>  
</configuration>
```
##### 使用客户端进行访问

```cs
using System;  
using System.Collections.Generic;  
using System.Linq;  
using System.Text;  
using GettingStartedClient.ServiceReference1;  
namespace GettingStartedClient  
{  
    class Program  
    {  
        static void Main(string[] args)  
        {  
            //Step 1: Create an instance of the WCF proxy.  
            CalculatorClient client = new CalculatorClient();  

            // Step 2: Call the service operations.  
            // Call the Add service operation.  
            double value1 = 100.00D;  
            double value2 = 15.99D;  
            double result = client.Add(value1, value2);  
            Console.WriteLine("Add({0},{1}) = {2}", value1, value2, result);  

            // Call the Subtract service operation.  
            value1 = 145.00D;  
            value2 = 76.54D;  
            result = client.Subtract(value1, value2);  
            Console.WriteLine("Subtract({0},{1}) = {2}", value1, value2, result);  

            // Call the Multiply service operation.  
            value1 = 9.00D;  
            value2 = 81.25D;  
            result = client.Multiply(value1, value2);  
            Console.WriteLine("Multiply({0},{1}) = {2}", value1, value2, result);  

            // Call the Divide service operation.  
            value1 = 22.00D;  
            value2 = 7.00D;  
            result = client.Divide(value1, value2);  
            Console.WriteLine("Divide({0},{1}) = {2}", value1, value2, result);  

            //Step 3: Closing the client gracefully closes the connection and cleans up resources.  
            client.Close();  
        }  
    }  
}  
```
