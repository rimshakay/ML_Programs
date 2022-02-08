function myFunc(){
    document.getElementById("display_box").innerHTML = '11';
    var pythonScriptPath='ML_Programs.RandomForestRegressor.simpleimputer';
    document.getElementById("display_box").innerHTML = '12';
    // // const PythonShell = require('python-shell');
    // document.getElementById("display_box").innerHTML = '13';
    // // var pyshell = new PythonShell(pythonScriptPath)
    // document.getElementById("display_box").innerHTML = '14';

    // pyshell.on('message', function (message) {
    //     // received a message sent from the Python script (a simple "print" statement)
    //     console.log(message);
    // });
    
    // // end the input stream and allow the process to exit
    // pyshell.end(function (err) {
    //     if (err){
    //         throw err;
    //     };
    
    //     console.log('finished');
    // });

    $.ajax({
        type: "POST",
        url: "/RandomForestRegressor/simpleimputer.py"
        // data: { param: text}
      }).done(function( o ) {
        document.getElementById("display_box").innerHTML = '13';
      });

    // var Model = require('web.Model');
    // var model  = new Model('my.model').call('checkSales').then(function(result){
    //     return result;
    // });

    
}