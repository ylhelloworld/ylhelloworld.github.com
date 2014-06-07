var home_app=angular.module('home_app',[]);
home_app.controller('home_controller', function ($scope) {
    $scope.page_name="Home Page";
    $scope.top_menu=[
                         {name:"Home",link:"#",id:"menu1",type:"active"},
                         {name:"Second",link:"#",id:"menu2",type:""},
                         {name:"Third",link:"#",id:"menu3",type:""},
                         {name:"Fouth",link:"#",id:"menu4",type:""},
                         {name:"Fifth",link:"#",id:"menu5",type:""},
                         {name:"Last",link:"#",id:"menu6",type:""} 
                    ];
    $scope.pager=[ 
                   {name:"Pre",link:"#"},
                   {name:"1",link:"test.htm"},
                   {name:"2",link:"#"},
                   {name:"3",link:"#"},
                   {name:"4",link:"#"},
                   {name:"5",link:"#"},
                   {name:"Next",link:"#"}
                 ];
    $scope.left_menu=[
                         {name:"First ipsum",link:"#",type:"active"},
                         {name:"Second ipsum",link:"#",type:""},
                         {name:"Third ipsum",link:"#",type:""},
                         {name:"Fouth ipsum",link:"#",type:""},
                         {name:"Fifth ipsum",link:"#",type:""},
                         {name:"Sixth ipsum",link:"#",type:""},
                         {name:"Seventh ipsum",link:"#",type:""},
                         {name:"Last ipsum",link:"#",type:""}
                     ]
    $scope.list = [
           {name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
           {  name: 'name',
           description: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit,  sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,  quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' }
    ]; 
});