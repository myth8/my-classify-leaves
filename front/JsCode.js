// by Chtiwi Malek on CODICODE.COM

function UploadPic() {
    var fileInput = document.getElementById('num_img');
    var file = fileInput.files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
    var dataURL = e.target.result;
    var base64Data = dataURL.replace(/^data:image\/(png|jpg);base64,/, "");
    // 在这里处理 base64Data
        $.ajax({
        type: 'POST',
        url: 'http://localhost:6767/recognize',
        data:JSON.stringify({imageData: base64Data}),
        contentType: 'application/json; charset=utf-8',
        dataType: 'html',
        success: function (msg) {
        var label = document.getElementById("result")
           label.textContent=msg;
        }
    });
    }
    reader.readAsDataURL(file);
}

