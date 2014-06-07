function  execute_method(param_type,param_method,param_input,call_back){
    $.getJSON("response.aspx",
               {type:param_type,method:param_method,input:JSON.stringify(param_input)},
               function(data){ 
                  call_back(data);
    }); 
}