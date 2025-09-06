from django.shortcuts import render
import web
import pymysql
import hashlib
import tempfile
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用 Agg 后端，不需要显示图形界面
import numpy as np
import math
import ast
import json
from io import BytesIO
from Dashboard.models import *
headers = {
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Origin': "*"
}
import copy

# from subFunctions import *
# Create your views here.

def sqlSelect(sql):
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='123456', db='photology')
    cur = conn.cursor()
    cur.execute(sql)
    sqlData = cur.fetchall()
    cur.close()
    conn.close()
    return sqlData

def sqlWrite(sql):
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='123456', db='photology')
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()
    conn.close()
    return

def photology(request):
    return render(request,'photology.html')

def datav(request):
    return render(request,'datav.html')

def sy(request):
    return render(request,'sy.html')

def test(request):
    return render(request,'test.html')

def a(request):
    return render(request,'a.html')

def a1(request):
    return render(request,'a1.html')

def a2(request):
    return render(request,'a2.html')

def a3(request):
    return render(request,'a3.html')

def a4(request):
    return render(request,'a4.html')

def a5(request):
    return render(request,'a5.html')

def a6(request):
    return render(request,'a6.html')

def a7(request):
    return render(request,'a7.html')

def a8(request):
    return render(request,'a8.html')

def a9(request):
    return render(request,'a9.html')

def b(request):
    return render(request,'b.html')

def b1(request):
    return render(request,'b1.html')

def b2(request):
    return render(request,'b2.html')

def b3(request):
    return render(request,'b3.html')

def b4(request):
    return render(request,'b4.html')

def b5(request):
    return render(request,'b5.html')

def b6(request):
    return render(request,'b6.html')

def b7(request):
    return render(request,'b7.html')

def b8(request):
    return render(request,'b8.html')

def b9(request):
    return render(request,'b9.html')

def b10(request):
    return render(request,'b10.html')

def b11(request):
    return render(request,'b11.html')

def b12(request):
    return render(request,'b12.html')

def b13(request):
    return render(request,'b13.html')

def b14(request):
    return render(request,'b14.html')

def c(request):
    return render(request,'c.html')

def c1(request):
    return render(request,'c1.html')

def c2(request):
    return render(request,'c2.html')

def c3(request):
    return render(request,'c3.html')

def c4(request):
    return render(request,'c4.html')

def c5(request):
    return render(request,'c5.html')

def c6(request):
    return render(request,'c6.html')

def c7(request):
    return render(request,'c7.html')

def c8(request):
    return render(request,'c8.html')

def c9(request):
    return render(request,'c9.html')

def c10(request):
    return render(request,'c10.html')

def c11(request):
    return render(request,'c11.html')

def c12(request):
    return render(request,'c12.html')

def c13(request):
    return render(request,'c13.html')

def c14(request):
    return render(request,'c14.html')

def c15(request):
    return render(request,'c15.html')

def c16(request):
    return render(request,'c16.html')

def c17(request):
    return render(request,'c17.html')

def c18(request):
    return render(request,'c18.html')
    
def rpw(request):
    if request.method=="GET":
        return render(request,"rpw.html",{"img_b64":" ","error":" "})
    else:
        try:
            t1 = float(request.POST.get("t1"))
            z=t1
            n1 = float(request.POST.get("n1"))
            n2 = float(request.POST.get("n2"))
            rL = float(request.POST.get("rL"))
            A = json.loads(request.POST.get("A", "[0]"))
            B = json.loads(request.POST.get("B", "[0]"))
            t1=t1*math.pi/180
            t2=math.asin(math.sin(t1)*n1/n2)
            v1=np.array([math.sin(t1)-math.cos(t1)])
            v2=np.array([math.sin(t2)-math.cos(t2)])  
            pA = np.array(A); pB = np.array(B)
            pA0 = pA - rL * v1  # 光线A初始点
            pB0 = pB - rL * v1  # 光线B初始点
            pA1 = pA + rL * v2  # 光线A结束点
            pB1 = pB + rL * v2  # 光线B结束点
            xA = np.array([pA0[0], pA[0], pA1[0]])
            yA = np.array([pA0[1], pA[1], pA1[1]])
            xB = np.array([pB0[0], pB[0], pB1[0]])
            yB = np.array([pB0[1], pB[1], pB1[1]])

            # 保存数据到数据库
            Rpw.objects.create(
                t1=z,
                n1=n1,
                n2=n2,
                rl=rL,
                a=A,
                b=B
            )

            # 绘制图形
            fig, ax = plt.subplots()
            ax.plot(xA, yA, 'k', label='ray A')  # 光线A
            ax.plot(xB, yB, 'r', label='ray B')  # 光线B
            ax.plot([-5, 5], [0, 0], ':g')
            ax.plot([0,0], [-5,5], ':g')
            ax.annotate('n1', xy=(3, 1), fontsize=16)
            ax.annotate('n2', xy=(3, -2), fontsize=16)
            ax.legend()
            
            # 保存图形
            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close(fig)  # 关闭图形，释放内存
            
            return render(request, "rpw.html", {"img_b64":img_b64, "error":""})
        except Exception as e:
            return render(request, "rpw.html", {"img_b64":"", "error":str(e)})

def rsw(request):
    if request.method=="GET":
        return render(request,"rsw.html",{"img_b64":" ","error":" "})
    else:
        try:
            th4 = float(request.POST.get("th4"))
            th3 = float(request.POST.get("th3"))
            lensR = float(request.POST.get("lensR"))
            th1 = float(request.POST.get("th1"))
            th2 = float(request.POST.get("th2"))
            A = request.POST.get("A")
            A = ast.literal_eval(A)  # 将输入字符串转换为列表
            A = np.array(A, dtype=float)  # 确保是数值类型的数组
            t = A / 180 * math.pi  # in arc
            lensR=2*lensR

            # 绘制图形
            fig, ax = plt.subplots()
            
            # 绘制光线
            for angle in t:
                # 计算入射光线的起点和终点
                x1 = np.array([th4, th1])
                y1 = np.array([0, np.tan(angle) * (th1 - th4)])
                
                # 计算透镜内部光线的起点和终点
                x2 = np.array([th1, th2])
                y2 = np.array([y1[1], y1[1] + np.tan(angle) * (th2 - th1)])
                
                # 计算出射光线的起点和终点
                x3 = np.array([th2, th3])
                y3 = np.array([y2[1], 0])
                
                # 绘制三段光线
                ax.plot(x1, y1, 'b-', linewidth=1)
                ax.plot(x2, y2, 'g-', linewidth=1)
                ax.plot(x3, y3, 'r-', linewidth=1)
            
            # 绘制透镜
            ax.axvline(x=th1, color='k', linestyle='--', alpha=0.5)
            ax.axvline(x=th2, color='k', linestyle='--', alpha=0.5)
            
            # 设置坐标轴范围和标签
            ax.set_xlim(th4-10, th3+10)
            ax.set_ylim(-50, 50)
            ax.set_xlabel('Position (mm)')
            ax.set_ylabel('Height (mm)')
            ax.grid(True, alpha=0.3)
            
            # 保存图形
            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close(fig)  # 关闭图形，释放内存
            
            return render(request, "rsw.html", {"img_b64":img_b64, "error":""})
        except Exception as e:
            return render(request, "rsw.html", {"img_b64":"", "error":str(e)})

def psfMTF(request):
    if request.method=="GET":
        return render(request,"psfMTF.html",{"img_b64":" ","img_b64_1":" ","img_b64_2":" ","error":" "})
    else:
        u0 = float(request.POST.get("u0"))
        v0 = float(request.POST.get("v0"))
        wd = float(request.POST.get("wd"))
        w040 = float(request.POST.get("w040"))
        w131 = float(request.POST.get("w131"))
        w222 = float(request.POST.get("w222"))
        w220 = float(request.POST.get("w220"))
        w311 = float(request.POST.get("w311"))
        if "submitBtn" in request.POST:
            Psfmtf.objects.create(
                u0=u0,
                v0=v0,
                w040=w040,
                w131=w131,
                w222=w222,
                w220=w220,
                w311=w311
            )
        def circ(r):
            return np.absolute(r) <= 1
        def rect(x):
            return np.absolute(x) <= 1 / 2

        def sedel_5(u0, v0, X, Y, wd, w040, w131, w222, w220, w311):
            beta = np.arctan2(v0, u0)  # 图像旋转角度
            u0r = np.sqrt(u0**2 + v0**2)  # 图像高度
            x_hat = X * np.cos(beta) + Y * np.sin(beta)  # 旋转网格
            y_hat = -X * np.sin(beta) + Y * np.cos(beta)
            
            rho2 = np.absolute(x_hat)**2 + np.absolute(y_hat)**2  # 径向长度
            
            # 赛德尔多项式
            w = (wd * rho2 +
                w040 * (rho2**2) +
                w131 * u0r * rho2 * x_hat +
                w222 * (u0r**2) * (x_hat**2) +
                w220 * (u0r**2) * rho2 +
                w311 * (u0r**3) * x_hat)
        
            return w
        # 定义物面
        M = 1024
        L = 1e-03
        du = L / M
        u = np.arange(-L / 2, (L / 2), du)
        v = u

        # 定义光学系统
        wave = 0.55e-6
        k = 2 * np.pi / wave
        Dxp = 20e-3
        wxp = Dxp / 2
        zxp=100e-3
        fnum = zxp / Dxp
        lz = wave * zxp
        twof0 = 1 / (wave * fnum)
        # 设置输入波像差系数
        wd = wd * wave
        w040 = w040 * wave
        w131 = w131 * wave
        w222 = w222 * wave
        w220 = w220 * wave
        w311 = w311 * wave

        # 图像频域坐标
        fu = np.arange(-1 / (2 * du), 1 / (2 * du), (1 / L))
        Fu, Fv = np.meshgrid(fu, fu)

        # 波前
        W = sedel_5(u0, v0, -lz * Fu / wxp, -lz * Fv / wxp, wd, w040, w131, w222, w220, w311)

        # 传递函数相位
        H = circ(np.sqrt(np.absolute(Fu)**2 + np.absolute(Fv)**2) * (lz / wxp)) * np.exp(-1j * k * W)
        H_a = np.angle(H)
        H_a[H_a==np.pi]==0

        # 绘制相干传递函数
        fig, ax = plt.subplots()
        ax.imshow(H_a, extent=[u[0], u[-1], v[0], v[-1]], cmap=plt.cm.gray)
        ax.set_title("coherent transfer function")
        ax.set_xlabel("u(m)")
        ax.set_ylabel("v(m)")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')


        # 点扩散函数
        h2 = np.absolute(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(H))))  # 点扩散函数
        fig1, ax1 = plt.subplots()
        ax1.imshow(h2 ** (1/2), extent=[u[0], u[-1], v[0], v[-1]], cmap=plt.cm.gray)
        ax1.set_title("point spread function")
        ax1.set_xlabel("x(m)")
        ax1.set_ylabel("v(m)")
        img1 = io.BytesIO()
        fig1.savefig(img1, format='png')
        img1.seek(0)
        img_b64_1 = base64.b64encode(img1.getvalue()).decode('utf-8')

        # MTF with aberration
        MTF = np.fft.fft2(np.fft.fftshift(h2))
        MTF = np.absolute(MTF / MTF[1, 1])
        MTF = np.fft.ifftshift(MTF)

        # MTF without aberration
        temp = fu / twof0
        temp_nor = temp / (np.absolute(temp)).max()
        fai = np.arccos(temp_nor)
        MTF_an = (2 / np.pi) * (fai - np.cos(fai) * np.sin(fai))

        # MTF curve
        fig4, ax4 = plt.subplots()
        ax4.plot(fu, MTF[int(M/2)+1, :], 'r', label='u方向MTF曲线')
        ax4.plot(fu, MTF[:, int(M/2)+1], 'g', label='v方向MTF曲线')
        ax4.plot(fu, MTF_an,'--',color= 'b', label='无像差MTF曲线')
        ax4.set_title("MTF curve")
        ax4.set_xlabel("f(cyc/m)")
        ax4.set_xlim(0, 150000)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel("Modulation")
        ax4.legend(loc='upper right')
        # 将图像保存为字节流并转换为 base64
        img2 = io.BytesIO()
        fig4.savefig(img2, format='png')
        img2.seek(0)
        img_b64_2 = base64.b64encode(img.getvalue()).decode('utf-8')
        return render(request,"psfMTF.html",{"img_b64":img_b64,"img_b64_1":img_b64_1,"img_b64_2":img_b64_2,"error":" "})
    
def a91(request):
    if request.method=="GET":
        return render(request,"a91.html",{"img_b64":" ","error":" "})
    else:
        c=float(request.POST.get("c"))
        b=float(request.POST.get("b"))
        d=int(request.POST.get("d"))
        if "submitBtn" in request.POST:
            A91.objects.create(
                c=c,
                b=b,
                d=d,
            )
        a = math.sqrt(b**2 + c**2)  # 椭圆方程
        x = np.arange(-40, 40, 0.1)  # x取值
        y = b * np.sqrt(1 - x**2 / a**2)  # y取值  # 画椭球面
        xl = np.arange(-40, 0.5)  # x取值
        yl = b * np.sqrt(1 - xl**2 / a**2)  # y取值
        fig, ax = plt.subplots()
        ax.plot(x, y, 'k', linewidth=4)
        for i in np.arange(0, d):  # 两点连线
            plt.plot([-40, xl[i]], [0, yl[i]], 'b')
            plt.plot([40, xl[i]], [0, yl[i]], 'b')
        ax.plot([-40, 40], [0, 0], '*k')  # 画两个焦点
        ax.plot([-40, 40], [0, 0], ':r')  # 画坐标轴
        ax.plot([0, 0], [-10, 40], ':r')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.xlim(-50, 50)
        plt.ylim(-15, 45)
        plt.title('Ideal elliptical mirror')
        plt.xlabel('x/mm')
        plt.ylabel('y/mm')
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
        return render(request,"a91.html",{"img_b64":img_b64,"error":" "})

# def a92(request):
#     if request.method=="GET":
#         return render(request,"a92.html",{"img_b64":" ","error":" "})
#     else:
#         def nrm(x):
#             return x/abs(x)
#         def coptpt(F,n1,v,G,n2,S,l):
#             return F+()
#         def angh(x):
#             return x
#         def rfr(i, ns, n1, n2):
#             i = nrm(i)
#             ns = nrm(ns)
#             if np.dot(i, ns) >= 0:
#                 n = ns
#             else:
#                 n = -ns
#             delta = 1 - ((n1/n2) ** 2) * (1 - np.dot(i, n) ** 2)
#             if delta > 0:
#                 outv = n1/n2 * i + (-np.dot(i, n) * n1/n2 + math.sqrt(delta)) * n
#             else:
#                 print('TIR happens')
#             return outv
        
#         def rfrnrm(i,r,n1,n2):
#             i=nrm(i)
#             r=nrm(r)
#             outv=(n1*i-n2*r)/np.linalg.norm(n1*i-n2*r)
#             return outv
        
#         def coptsl(F, n1, v1, Q, n2, v2, S):
#             v1 = nrm(v1)
#             v2 = nrm(v2)
#             outp = (S - n2 * np.dot(Q - F, v2)) / (n1 - n2 * np.dot(v1, v2)) * v1 + F
#             return outp
        
#         def ccoptpt(F,n1,v,G,n2,S):
#             outp=coptpt(F,n1,v,G,n2,S,1)
#             return outp
        
#         n1=float(request.POST.get("n1"))
#         n2=float(request.POST.get("n2"))
#         theta=float(request.POST.get("n2"))
#         theta=theta/180 * math.pi  # 光束入射角，根据光束入射角计算折射角
#         nR1=np.array([math.sin(theta), -math.cos(theta)])
#         nR2=np.array([-math.sin(theta), -math.cos(theta)])
#         wFR1=np.array([math.cos(theta), math.sin(theta)])
#         wFR2=np.array([math.cos(theta), -math.sin(theta)])
#         PointC=- nR1
#         PointF=- nR2
#         PointE1=np.array([-2, -15])
#         PointE2=np.array([2, -15])

#         # 2. 从 C 点出发经过 O 计算透镜下表面点坐标
#         L_OD=2.03  # OD的长度，单位
#         PointO=np.array([0, 0])  # 透镜上表面顶点
#         nPointO=np.array([0, 1])  # 透镜上表面顶点处都有一个法向量
#         rePointO=rfr(nR1, nPointO, n1, n2)  # 由(R1 波前)→O根据入射方向、折射方向、折射率求入射点法线
#         PointD=PointO+L_OD * nrm(rePointO)  # 透镜下表面 D 点坐标
#         rePointD=nrm(PointE2 - PointD)  # 透镜下表面顶点 D 点光线的折射方向
#         nPointD=rfrnrm(rePointO, rePointD, n2, n1)  # D 处法线 rfrnrm 根据入射方向、折射方向、折射率求入射点法线

#         PointE=np.array([-PointD[0], PointD[1]])  # D 关于 y 轴对称的点 E 的坐标

#         # 3. 总光程
#         OPL=n1 * np.linalg.norm(PointC - PointO) + n2 * np.linalg.norm(PointO - PointD) + np.linalg.norm(PointD - PointE2)

#         # 4. 计算透镜右半边上下表面的点坐标
#         samplePoints=50  # ED 之间采样点的个数
#         S1 = np.zeros(shape=(samplePoints+1, 4))  # 保存透镜上表面的点坐标
#         S2 = np.zeros(shape=(samplePoints+1, 4))  # 保存透镜下表面的点坐标

#         # 下表面 ED 拟合函数：y=a+bx^2  dy/dx=2bx=tan(alpha+pi/2) 该公式求解 b
#         alpha = angh(nPointD)  # 求 rePointD 相对于水平线的角度
#         b = math.tan(alpha + math.pi/2) / 2 / PointD[0]
#         a = PointD[1] - b * PointD[0] * 2

#         # 计算一个循环下上表面点
#         i=0
#         for x in np.arange(0, 1/samplePoints+1, 1/samplePoints):
#             # 1) 使用拟合函数生成 ED 之间的一段点
#             S2Pointx = x * PointD[0] + (1 - x) * PointE[0]  # 取 ED 之间的某一个点 S2p 的横坐标
#             S2Pointy = f(a, b, S2Pointx)
#             S2Point = np.array([S2Pointx, S2Pointy])
#             S2Point = np.array([S2Pointx, S2Pointy])
#             nS2Point = drf(nrm(np.array([f(a, b, S2Pointx), -1])))
#             reS2Point = rfr(nrm(np.array([f(a, b, S2Pointx), -1])), nS2Point, n1, n2)  # 光线 E1→S2p 在 S2p 点处的折射向量
#             # 保存 S2Point 坐标
#             S2[i, 0:2] = S2Point
#             # 保存 S2Point 处的法向量
#         # 2) 由 ED 之间的一段点生成 OG 之间的一段点
#         OPLx = OPL - np.linalg.norm(PointE1 - S2Point)  # S2Point→(R2 波前)的光程
#         S1Point = coptsl(S2Point, n2, reS2Point, PointF, n1, -nR2, OPLx)  # 与 S2Point 对应的点 S1Point 的坐标
#         nS1Point = rfrmm(reS2Point, -nR2, n2, n1)  # S1Point 处法线 rfrmm 根据入射方向、折射方向、折射率求入射点法线
#         S1[i, 0:2] = S1Point  # 保存 S1Point 坐标
#         S1[i, 2:4] = nS1Point  # 保存 S1Point 处的法向量
#         i = i + 1

#         # 计算接下来的 N 段点
#         N = 21
#         for j in range(N):
#             for i in range(j * samplePoints + 1, (j + 1) * samplePoints + 1):
#                 # 1) 求第 j 段的 S2i
#                 # Sli 与 R1 波前的交点 L
#                 R1_temp_Point = isl(S1[i, 0:2], -nR1, PointC, wFR1)  # S1i 与 R1 波前的交点 L
#                 OPLi = OPL - np.linalg.norm(S1[i, 0:2] - R1_temp_Point)  # S1i→E2 的光程
#                 reS1i = rfr(nR1, S1[i, 2:4], n1, n2)  # 光线 S1i→E2 在 S1i 处折射方向向量
#                 S2i = coptsl(S1[i, 0:2], n2, reS1i, PointE2, n1, OPLi)  # 与 S1i 对应的 S2i 的坐标
#                 nS2i = rfrmm(nrm(S2i - S1[i, 0:2]), nrm(PointE2 - S2i), n2, n1)  # S2i 处的法向量
#                 S2 = np.row_stack((S2, np.concatenate((S2i, nS2i))))  # 将 S2i 的坐标，以及法向量追加在 S2 中

#                 # 2) 求第 i 段的 S1i
#                 OPLi = OPL - np.linalg.norm(PointE1 - S2[i + samplePoints, 0:2])  # S2i→(R1 波前)的光程
#                 reS2i = rfr(nrm(S2[i + samplePoints, 0:2] - PointE1), S2[i + samplePoints, 2:4], n1, n2)  # 光线 E1→S2i 在 S2i 处折射方向向量
#                 S1i = coptsl(S2[i + samplePoints, 0:2], n2, reS2i, PointF, n1, -nR2, OPLi)  # 与 S2i 对应的新的 S1i 的坐标
#                 nS1i = rfrmm(reS2i, nrm(-nR2), n2, n1)  # S1i 处的法向量
#                 S1 = np.row_stack((S1, np.concatenate((S1i, nS1i))))  # 将 S1i 的坐标，以及法向量追加在 S1 中
#         MS1=copy.copy(S1)
#         MS1[:, 0]=-MS1[:, 0]  # 将 S1 赋值给 MS1 并关于 y 轴对称
#         MS1=np.row_stack((MS1, S1))  # 将 MS1 和 S1 放在一个矩阵 MS1 中

#         MS2=copy.copy(S2)
#         MS2[:, 0]=-MS2[:, 0]  # 将 S2 赋值给 MS2 并关于 y 轴对称
#         MS2=np.row_stack((MS2, S2))  # 将 MS2 和 S2 放在一个矩阵 MS2 中
#         fig, ax = plt.subplots()
#         # 5. 绘制曲线点
#         plt.title(u'Symmetry of lens profiles for +/- 8 deg')
#         plt.plot(S1[:, 0], S1[:, 1], 'xr')
#         plt.plot(S2[:, 0], S2[:, 1], 'xb')
#         plt.title(u'Symmetry of lens profiles for +/- 8 deg')
#         plt.plot(MS1[:, 0], MS1[:, 1], 'xr')
#         plt.plot(MS2[:, 0], MS2[:, 1], 'xb')
#         img = io.BytesIO()
#         fig.savefig(img, format='png')
#         img.seek(0)
#         img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
#         return render(request,"a92.html",{"img_b64":img_b64,"error":" "})

def c21(request):
    if request.method == "GET":
        return render(request, "c21.html", {"img_b64": " ","error": " "})

    try:
        wavelength = float(request.POST.get("wavelength"))
        theta = float(request.POST.get("theta"))
        if "submitBtn" in request.POST:
            C21.objects.create(
                wavelength=wavelength,
                theta=theta
            )
        wavelength = wavelength * 1e-6  # 单位: m
        k = 2 * np.pi / wavelength  # 波数
        xdir = np.cos(theta * np.pi / 180)
        ydir = np.sin(theta * np.pi / 180)
        kx, ky, kz = k * xdir, k * ydir, 0.0  # 计算k的各分量

        # 生成空间网格
        x = np.linspace(-2 * wavelength, 2 * wavelength, 200)
        y = np.linspace(-2 * wavelength, 2 * wavelength, 200)
        X, Y = np.meshgrid(x, y)

        # 计算波的传播
        A = 1.0  # 振幅
        P = np.exp(1j * (kx * X + ky * Y))  # 传播函数
        U = A * P  # 波场

        # 计算强度和相位
        intensity = np.abs(U) ** 2  # 强度
        phase = np.angle(U)  # 相位

        # 创建图形并绘制图像
        fig, ax = plt.subplots(1, 2)
        mag = ax[0].imshow(intensity, cmap=plt.cm.gray, origin='lower')
        ax[0].set_title('Intensity')
        ax[0].axis('off')
        fig.colorbar(mag, ax=ax[0])

        ang = ax[1].imshow(phase, cmap=plt.cm.bwr, origin='lower')
        ax[1].set_title('Phase')
        ax[1].axis('off')
        fig.colorbar(ang, ax=ax[1])

        # 将图像保存到内存中并转换为base64编码
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')

        # 返回渲染的页面，并传递base64编码的图片和没有错误信息
        return render(request, "c21.html", {"img_b64": img_b64,"error": " "})

    except Exception as e:
        # 如果出现错误，显示错误信息
        return render(request, "c21.html", {"img_b64": " ","error": str(e)})
    
def c22(request):
    if request.method == "GET":
        # 初始页面，返回空白图片和没有错误信息
        return render(request, "c22.html", {"img_b64": " ","error": " "})

    try:
        # 获取用户输入的波长和源位置
        wavelength = float(request.POST.get("wavelength", 0.5)) * math.e-6  # 单位: m, 默认值为0.5微米
        xc = float(request.POST.get("xc"))  # 源位置的x坐标
        yc = float(request.POST.get("yc"))  # 源位置的y坐标
        if "submitBtn" in request.POST:
            C22.objects.create(
                wavelength=request.POST.get("wavelength"),
                xc=xc,
                yc=yc,
            )
        # 计算波数
        k = 2 * np.pi / wavelength

        # 生成空间网格
        x = np.linspace(-2 * wavelength, 2 * wavelength, 200)
        y = np.linspace(-2 * wavelength, 2 * wavelength, 200)
        X, Y = np.meshgrid(x, y)

        # 计算源到每个点的距离R
        R = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

        # 计算振幅和传播函数
        A = 1.0 / R  # 振幅，随距离R变化
        P = np.exp(1j * k * R)  # 传播函数

        # 计算波场
        U = A * P

        # 计算强度和相位
        intensity = np.abs(U) ** 2  # 强度
        phase = np.angle(U)  # 相位

        # 创建图形并绘制图像
        fig, ax = plt.subplots(1, 2)
        mag = ax[0].imshow(np.log(intensity), cmap=plt.cm.gray, origin='lower')  # 对强度进行对数处理
        ax[0].set_title('Intensity')
        ax[0].axis('off')
        fig.colorbar(mag, ax=ax[0])

        ang = ax[1].imshow(phase, cmap=plt.cm.bwr, origin='lower')
        ax[1].set_title('Phase')
        ax[1].axis('off')
        fig.colorbar(ang, ax=ax[1])

        # 将图像保存到内存中并转换为base64编码
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')

        # 返回渲染的页面，并传递base64编码的图片和没有错误信息
        return render(request, "c22.html", {"img_b64": img_b64,"error": " "})

    except Exception as e:
        # 如果出现错误，显示错误信息
        return render(request, "c22.html", {"img_b64": " ", "error": str(e)})
    
def syrecord(request):
    rpwData=Rpw.objects.filter()[0:15]
    a91Data=A91.objects.filter()[0:15]
    c21Data=C21.objects.filter()[0:15]
    c22Data=C22.objects.filter()[0:15]
    rswData=Rsw.objects.filter()[0:15]
    psfmtfData=Psfmtf.objects.filter()[0:15]
    return render(request,'syrecord.html',{"rpwData":rpwData,"rswData":rswData,"a91Data":a91Data,"c21Data":c21Data,"c22Data":c22Data,"psfmtfData":psfmtfData})