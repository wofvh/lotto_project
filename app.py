<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chat Room</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
    <script type="text/javascript" src="//cdnjs.cloudflare.com/ajax/libs/socket.io/1.4.8/socket.io.min.js"></script>
</head>
<body>
    <script type="text/javascript">
        $(document).ready(function(){
            var sock = io.connect('http://127.0.0.1:8080');
            sock.on('connect', function(){
                var connect_string = 'new_connect';
                sock.send(connect_string);
            });

            sock.on('hello', function(msg){
                $('#messages').append('<li>' +'>>Hello :'+ msg + '</li>');
                console.log('Received Hello Message');
            });

            sock.on('message', function(msg){
                if(msg.type === 'normal'){
                    $('#messages').append('>> '+msg.message+'<br>');
                }else{
                    $('#messages').append('<li>' + msg.message + '</li>');
                }
                console.log('Received Message : '+msg.type);
            });

            $('#sendbutton').on('click', function(){
                sock.send($('#myMessage').val());
                $('#myMessage').val('');
            });

            $('#myMessage').on('keyup', function(event){
                if (event.key === "Enter") {
                    $('#sendbutton').click();
                }
            });
        });
    </script>
    <ul id="messages"></ul>
    <input type="text" id="myMessage">
    <button id="sendbutton">Send</button>
</body>
</html>
