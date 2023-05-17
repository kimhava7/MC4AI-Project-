import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data import df


st.title('BẢNG ĐIỂM PY4AI 09/2022')
tab1,tab2,tab3,tab4 = st.tabs(['Danh sách','Biểu đồ','Phân nhóm','Phân loại'])

with tab1:
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.write('Giới tính')
        check1 = st.checkbox('Nam')
        check2 = st.checkbox('Nữ')

    with col2:
        choice = st.radio('Khối lớp',['Tất cả','Lớp 10','Lớp 11','Lớp 12'])

    with col3:
        room = st.selectbox('Phòng',['Tất cả','A114','A115'])

    with col4:
        times = st.multiselect('Buổi',['Sáng','Chiều','Tất cả'])
    
    with st.container():
        st.write('Lớp chuyên')
        co1,co2,co3,co4,co5 = st.columns(5,gap='small')
        with co1:
            cv = st.checkbox('Văn')
            ct = st.checkbox('Toán')

        with co2:
            cl = st.checkbox('Lý')
            ch = st.checkbox('Hóa')

        with co3:
            ca = st.checkbox('Anh')
            cti = st.checkbox('Tin')
            
        with co4:
            sd = st.checkbox('Sử Địa')
            tn = st.checkbox('Trung Nhật')

        with co5:
            thsn = st.checkbox('TH/SN')
            kh = st.checkbox('Khác')

    gender = []
    if check1==True:
        gender.append('M')
    
    if check2==True:
        gender.append('F')

    data = df[df['GENDER'].isin(gender)]
    
    if choice == 'Tất cả':
        data1 = data
    elif choice =='Lớp 10':
        data1 = data[data['CLASS'].str.contains('10')]
    elif choice == 'Lớp 11':
        data1 = data[data['CLASS'].str.contains('11')]
    else:
        data1 = data[data['CLASS'].str.contains('12')]
        
    if room == 'A114':
        data2 = data1[data1['PYTHON-CLASS'].str.contains('114')]
    elif room == 'A115':
        data2 = data1[data1['PYTHON-CLASS'].str.contains('115')]
    else:
        data2 = data1

    if times ==[ 'Sáng']:
        data3 = data2[data2['PYTHON-CLASS'].str.endswith('S')]
    elif times == ['Chiều']:
        data3 = data2[data2['PYTHON-CLASS'].str.endswith('C')]
    else:
        data3 = data2
    

    cla = []
    if cv == True:
        cla.append('Chuyên Văn')
    if ct == True:
        cla.append('Chuyên Toán')
    if cl == True:
        cla.append('Chuyên Lí')
    if ch == True:
        cla.append('Chuyên Hóa')
    if cti == True:
        cla.append('Chuyên Tin')
    if ca == True:
        cla.append('Chuyên Anh')
    if sd == True:
        cla.append('Sử Địa')
    if tn == True:
        cla.append('Trung Nhật')
    if thsn == True:
        cla.append('Tích hợp')
        cla.append('Song ngữ')
    if kh == True:
        cla.append('Khác')
    data4 = data3[data3['CLASS-GROUP'].isin(cla)]
    
    if len(data4)>0:
        man = len(data4[data4['GENDER']=='M'])
        fe = len(data4[data4['GENDER']=='F'])
        st.write('Số HS:',str(len(data4['GENDER'])),'(',str(man),'nam,',str(fe),'nữ)')    
        gpama = max(data4['GPA'])
        gpami = min(data4['GPA'])
        gpame = np.mean(data4['GPA'])
        st.write('GPA: cao nhất ',round(gpama,1),',thấp nhất',round(gpami,1),',trung bình',round(gpame,1))
        st.dataframe(data4)
    else:
        st.write('Số HS:',0,'(',str(0),'nam,',str(0),'nữ)')
        st.write('GPA: cao nhất ',0,',thấp nhất',0,',trung bình',0)
        st.dataframe(data4)


with tab2:
    S_list = np.array(["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"])
    def draw_pie(df,attribute):
        return px.pie(df,names = attribute)

    def python_class():
        python_class = df["PYTHON-CLASS"]
        return python_class
    def class_group():
        class_group = df["CLASS-GROUP"]
        return class_group
    def gender():
        gender = df["GENDER"]
        return gender
    
    filter = df.columns[4:14]
    df1 = df[filter].values
    mean_session = np.mean(df1,axis = 0)
    data1 = {"name":S_list,
            "mean":mean_session

    }
    mean_df = pd.DataFrame(data1)
    def draw_box(attribute,session,is_gender):
        if is_gender == False:
            return px.box(df,x = attribute ,y = df[session])
        else:
            return px.box(df,x = attribute ,y = df[session],color = gender())
  

    def draw_histogram(attribute,session,is_GPA = False):
        if is_GPA == False:
            return 
        else:
            return px.histogram(mean_df,x = mean_df["name"],y = mean_df["mean"])
    Nhan_xet = {
    
    "S1":
        {
        "Box1":
        """
        * Học sinh ở lớp 115-Chiều và 114-Sáng cao hơn so với các lớp khác
        * Học sinh ở lớp 114-Chiều có sự phân bố đều về điểm
        * Nữ sinh ở các lớp phần lớn chiếm số điểm cao hơn, trong khi Nam phân bố đều về điểm số
        """,
        "Box2":
        """
        * Các lớp Tích hợp/Song ngữ có sự phân bố đều về điểm
        * Lớp Chuyên Toán,Sử Địa,Chuyên Hoá và Chuyên Văn có số điểm khá cao, do đó các lớp học tốt hơn
        * Lớp Chuyên Anh khá tệ khi có phần lớn học sinh dưới điểm 6""",
        "Hist3":
        "* Số điểm Chuyên Toán giành được cao nhất so với các lớp còn lại"
        },
    "S2":
    {
        "Box1":
        """
        * Nhìn chung học sinh Nữ học tốt hơn học sinh Nam,lớp 114 học tốt hơn lớp 115""",
        "Box2":
        """

        * Chuyên Tin,Chuyên Toán và Chuyên Văn học tốt nhất
        * Chuyên Anh có sự phân bố điểm đồng đều từ 0-10
        * Trung Nhật học có phần kém hơn so với các lớp còn lại
        """,
        "Hist3":
        """
        
        """


    },
    "S3":
    {
        "Box1":
        """
        * Nhìn chung học sinh Nữ học tốt hơn học sinh Nam,lớp 114 học tốt hơn lớp 115""",
        "Box2":
        """
        * Chuyên Toán/ Sử Địa/ Hóa/Lý/Tin học tốt nhất và có số điểm rất cao từ >8 - 10
        * Chuyên Văn/Trung Nhật có số điểm khá thấp, học kém hơn các lớp còn lại
        """,
        "Hist3":
        """
        
        """


    },
    "S4":
    {
        "Box1":
        """
        * 114-S có kết quả cao nhất trong 4 lớp,học sinh Nam phần lớn học tốt hơn học sinh Nữ(trừ lớp 114-C)""",
        "Box2":
        """
        * Chuyên Toán/Hóa/Văn/Lý/Tin có kết quả tốt nhất
        * Chuyên Anh/Trung Nhật/Tích hợp/Song ngữ có kết quả kém nhất
        * Sử Địa có sự phân bố điểm đồng đều và đa dạng 
        """,
        "Hist3":
        """
        
        """
        },
        "S5":
    {
        "Box1":
        """
        * Lớp 114-S và 115-C có kết quả khá cao,học tốt hơn so với các lớp khác
        * 114-C và 115-S có kết quả đa dạng và phân bố đồng đều từ 0-9
        * Học sinh Nữ học tốt hơn học sinh Nam ở Session này

    """,
        "Box2":
        """
        * Chuyên Toán/Tin/Lý đạt kết quả tốt nhất
        * Chuyên Anh và Trung Nhật học kém hơn trong Session này
        """,
        "Hist3":
        """
        
        """
        },
    "S6":
    {
        "Box1":
        """
        * Ở điểm giữa kì , học sinh Nam làm tốt hơn so với học sinh Nữ, lớp 114 có kết quả cao hơn lớp 115

    """,
        "Box2":
        """
        * Chuyên Toán/Tin/Lý đạt kết quả tốt nhất
        * Tích hợp/Song ngữ có kết quả kém nhất
        * Nhưng nhìn chung, số điểm của các lớp khá tốt
        
        """,
        "Hist3":
        """
        
        """
        },
        "S7":
    {
        "Box1":
        """
        * Lớp Sáng có kết quả tốt hơn so với Lớp Chiều,học sinh Nữ đạt kết quả tốt hơn học sinh Nam
        * Nhìn chung, điểm Session này có sự phân bố đồng đều và đa dạng



    """,
        "Box2":
        """
        * Chuyên Tin/Lý/Hóa và các lớp không chuyên đạt kết quả cao
        * Sử Địa/Trung Nhật/Tích hợp/Song ngữ đạt kết quả thấp nhất

        """,
        "Hist3":
        """
        
        """
        },
        "S8":
    {
        "Box1":
        """
        * Lớp Sáng có kết quả tốt hơn so với Lớp Chiều,học sinh Nữ đạt kết quả tốt hơn học sinh Nam
        * Nhìn chung, điểm Session này có sự phân bố đồng đều và đa dạng



    """,
        "Box2":
        """
        * Chuyên Tin/Lý/Hóa/Sử Địa và các lớp không chuyên đạt kết quả cao
        * Trung Nhật/Tích hợp/Song ngữ đạt kết quả kém
        * Lớp Chuyên Anh có sự phân bố điểm số từ 0-10

        """,
        "Hist3":
        """
        
        """
        },
        "S9":
    {
        "Box1":
        """
        * Nhìn chung, điểm Session này có sự phân bố đồng đều và đa dạng,trong đó học sinh Nam có kết quả cao hơn học sinh Nữ,Lớp 114 có kết quả tốt hơn lớp 115
        * Kết luận: Session này khá khó do tất cả các lớp đều phân bố điểm từ 0-10, trong đó lớp 115 có kết quả kém hơn so với lớp 114



    """,
        "Box2":
        """

        * Trung Nhật/Tích hợp/Song ngữ đạt kết quả cực kì tệ, Sử Địa và Chuyên Hoá có kết quả phần lớn dưới điểm 6
        * Các lớp đều có sự phân hoá đồng đều từ 0-10, chứng tỏ rằng Session này khá khó đối với học sinh
        """,
        "Hist3":
        """
        
        """
        },
        "S10":
    {
        "Box1":
        """
        * Nhìn chung, điểm Session này khá tốt
        * Lớp 114 học tốt hơn so với lớp 115, học sinh Nam đạt kết quả cao hơn học sinh Nữ

    """,
        "Box2":
        """

        * Chuyên Tin/Toán/Lý đạt kết quả tốt nhất
        * Sử Địa/Tích hợp/Song ngữ/Trung Nhật đạt kết quả kém
        * Nhìn chung, điểm Session này khá tốt 
        """,
        "Hist3":
        """
        
        """
        },
    }
    df2 = df["CLASS-GROUP"]
    print(df2)
    # print(df2.unique())
    def mean_class(df2,session):
        mean_class_list = []
        
        for subject in df2.unique():
            
            df3 = df[df["CLASS-GROUP"].str.contains(subject)]
            
            male = df3[df3["GENDER"].str.contains("M")]
            female = df3[df3["GENDER"].str.contains("F")]
            male_class = male[session]
            female_class = female[session]
                # _list.append(np.mean(_class))

            mean_class_list.append([np.mean(male_class),np.mean(female_class)])
        return np.array(mean_class_list)

    def histogram3(session,mean_class):
        data4 ={
        "GENDER" : ["M","F"]
        }
        for cls in range(len(mean_class)):
            data4[df2.unique()[cls]] = mean_class[cls]
        _df = pd.DataFrame(data4)

        for cls_ in df2.unique():
            _df[cls_].fillna(0,inplace = True)
        male = _df.loc[0]
        female = _df.loc[1]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(histfunc = "sum",y= female.values[1:], x=df2.unique(), name=female["GENDER"]))
        fig.add_trace(go.Histogram(histfunc = "sum",y= male.values[1:], x=df2.unique(), name=male["GENDER"]))

        return fig





    #begin designing
    tab5,tab6 = st.tabs(["Số lượng học sinh","Điểm"])
    with tab5:
        st.plotly_chart(draw_pie(df,python_class()))
        st.success("Sự phân bố học sinh ở các buổi học tương đối đều, cho thấy sự sắp xếp,tổ chức buổi học hợp lý.")

        st.plotly_chart(draw_pie(df,class_group()))
        st.success("Lớp Chuyên Toán chiếm đa số, cho thấy lớp Chuyên Toán hứng thú với việc học lập trình AI hơn so với các lớp còn lại.")
        st.plotly_chart(draw_pie(df,gender()))
        st.success("Nam chiếm đa số, cho thấy Nam có sự quan tâm về AI hơn Nữ")
    with tab6:
        Chose_Session= st.radio("Điểm từng session",("S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","GPA"),horizontal = True)

        if Chose_Session in S_list:
            st.plotly_chart(draw_box(python_class(),Chose_Session,True))
            st.success(Nhan_xet[Chose_Session]["Box1"])
            

            st.plotly_chart(draw_box(class_group(),Chose_Session,False))
            st.success(Nhan_xet[Chose_Session]["Box2"])
            mean_class = mean_class(df2,Chose_Session)
            st.plotly_chart(histogram3(Chose_Session,mean_class))
            # st.success(Nhan_xet[Chose_Session]["Hist3"])
        elif Chose_Session == "GPA":
            st.plotly_chart(draw_box(python_class(),Chose_Session,True))
            st.success(
            """
            * Lớp 114 học tốt hơn so với lớp 115, học sinh Nam học tốt hơn so với học sinh Nữ
            * Nhìn chung, GPA của học sinh khá tốt
            """
            )
            st.plotly_chart(draw_box(class_group(),Chose_Session,False))
            st.success(
            """
            * Chuyên Tin học tốt nhất
            * Tích hợp/Song ngữ/Trung Nhật có kết quả kém hơn so với các lớp còn lại
            """
            )
            mean_class = mean_class(df2,Chose_Session)
            st.plotly_chart(histogram3(Chose_Session,mean_class))
            st.plotly_chart(draw_histogram(class_group(),Chose_Session,is_GPA = True))


with tab3:
    slider = st.slider('Số nhóm',2,5,3)
    
    def y(n):
        X = df[['S6','S10','S-AVG']].to_numpy()
        kmeans = KMeans(n_clusters=n,n_init='auto')
        kmeans.fit(X)
        y = kmeans.labels_
        return y

    def write(m):
        for i in range(m):
            data = df[y==i]
            gpamax = np.max(data['GPA'])
            gpamin = np.min(data['GPA'])
            gpamean = np.mean(data['GPA'])
            st.write('Nhóm '+str(i+1),': ','GPA cao nhất ',gpamax,', thấp nhất ',gpamin,',trung bình ',round(gpamean,2))
            st.dataframe(df[y==i][['NAME','CLASS','GPA']])

    X = df[['S6','S10','S-AVG']].to_numpy()
    if slider == 3:
        y = y(3)
        fig = go.Figure(
            data=[
                go.Scatter3d(x=X[y==0,0],y=X[y==0,1],z=X[y==0,2],mode='markers'),
                go.Scatter3d(x=X[y==1,0],y=X[y==1,1],z=X[y==1,2],mode='markers'),
                go.Scatter3d(x=X[y==2,0],y=X[y==2,1],z=X[y==2,2],mode='markers')
            ]
        )
        st.plotly_chart(fig)
        write(3)
        
    elif slider == 2:
        y = y(2)
        fig = go.Figure(
            data=[
                go.Scatter3d(x=X[y==0,0],y=X[y==0,1],z=X[y==0,2],mode='markers'),
                go.Scatter3d(x=X[y==1,0],y=X[y==1,1],z=X[y==1,2],mode='markers')
            ]
        )
        st.plotly_chart(fig)
        write(2)

    elif slider == 4:
        y = y(4)
        fig = go.Figure(
            data=[
                go.Scatter3d(x=X[y==0,0],y=X[y==0,1],z=X[y==0,2],mode='markers'),
                go.Scatter3d(x=X[y==1,0],y=X[y==1,1],z=X[y==1,2],mode='markers'),
                go.Scatter3d(x=X[y==2,0],y=X[y==2,1],z=X[y==2,2],mode='markers'),
                go.Scatter3d(x=X[y==3,0],y=X[y==3,1],z=X[y==3,2],mode='markers')
            ]
        )
        st.plotly_chart(fig)
        write(4)

    else:
        y = y(5)
        fig = go.Figure(
            data=[
                go.Scatter3d(x=X[y==0,0],y=X[y==0,1],z=X[y==0,2],mode='markers'),
                go.Scatter3d(x=X[y==1,0],y=X[y==1,1],z=X[y==1,2],mode='markers'),
                go.Scatter3d(x=X[y==2,0],y=X[y==2,1],z=X[y==2,2],mode='markers'),
                go.Scatter3d(x=X[y==3,0],y=X[y==3,1],z=X[y==3,2],mode='markers'),
                go.Scatter3d(x=X[y==4,0],y=X[y==4,1],z=X[y==4,2],mode='markers')
            ]
        )
        st.plotly_chart(fig)
        write(5)

with tab4:
    radio= st.radio('Số đặc trưng',(2,3),horizontal=True)

    result_mapping = {
        'Đ':0,
        'KĐ':1
    }
    y = df['RESULT'].map(result_mapping).to_numpy()

    if radio == 3:
        X = df[['S6','S10','S-AVG']]
        model = LogisticRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        Score = model.score(X, y)
        weights = model.coef_[0]
        bias = model.intercept_[0]

        w1, w2, w3 = weights
        b = bias

        x_new = np.linspace(2.5,8,100)
        y_new = np.linspace(2.5,8,100)
        xx, yy = np.meshgrid(x_new, y_new)
        xy = np.c_[xx.ravel(), yy.ravel()]
        z = -(w1*xy[:,0]+w2*xy[:,1]+b)/w3
        z = z.reshape(xx.shape)

        fig = go.Figure(
            data=[
                go.Scatter3d(x=X[y==0]['S6'],y=X[y==0]['S10'],z=X[y==0]['S-AVG'],mode='markers'),
                go.Scatter3d(x=X[y==1]['S6'],y=X[y==1]['S10'],z=X[y==1]['S-AVG'],mode='markers'),
                go.Surface(x=x_new, y=y_new, z=z)
            ]
        )
        
        st.plotly_chart(fig)
        st.write('Score ',round(Score,2))

    else:
        
        X = df[['S-AVG','S6']].to_numpy()
        # Creat and Fit model
        model = LogisticRegression()
        model.fit(X, y)

        # Score
        Score = model.score(X,y)

        # Lấy ra bộ trọng số
        weights = model.coef_[0]
        bias = model.intercept_[0]
        w1, w2 = weights

        x1 = np.linspace(1.5,9.5,1000)
        x2 = (-w1*x1-bias)/w2

        fig,ax = plt.subplots()
        ax.scatter(X[y==0,0],X[y==0,1],color='blue')
        ax.scatter(X[y==1,0],X[y==1,1],color='violet')
        ax.plot(x1,x2,color='red')
        ax.set_xlabel('S-AVG')
        ax.set_ylabel('S6')
        st.pyplot(fig)
        st.write('Score ',round(Score,2))