// 获取按钮元素
var button = document.getElementById('Single_Step_Button');

// 绑定点击事件
button.addEventListener('click', function () {
  // 全局变量cnt+1 并重新渲染index.html界面
  fetch('/increment_cnt', {
    method: 'POST'
  })
    .then(response => {
      if (response.ok) {
        // 请求成功后刷新页面
        window.location.reload();
      } else {
        console.error('Failed to increment index:', response.statusText);
      }
    })
    .catch(error => {
      console.error('Error incrementing index:', error);
    });
});


