from flask import Flask, request,Response # 导入 request
import json
# from waitress import serve # 可以注释掉或删掉 waitress 的 import

app=Flask(__name__)

@app.route('/demo')
def demo():
    name=request.args.get('name','guest')
    return f'hello,{name}!'

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        user_name=request.form.get('username')
        password=request.form.get('password')
        if user_name and password:
            return '登陆成功，baby！'
        else:
            return '密码或者账号没有填写'
    else: # GET 请求时，显示一个简单的登录表单
        return '''
            <form method="post">
                Username: <input type="text" name="username"><br>
                Password: <input type="password" name="password"><br>
                <input type="submit" value="Login">
            </form>
        '''
    
@app.route('/custom')
def custom():
    data={'status':200,'damn':'what can i say'}
    return Response(
        json.dumps(data),
        mimetype='application/json'
    )
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=9333,debug=True) # 恢复使用 app.run, 保持 debug=False
    # serve(app, host='0.0.0.0', port=9333) # 注释掉 waitress
