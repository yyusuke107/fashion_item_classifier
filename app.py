#以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict


st.set_option("deprecation.showfileUploaderEncoding",False)

st.sidebar.title("画像認識ファイル")
st.sidebar.write("オリジナルの画像認識モデルを使って何の画像かを判定します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。", ("画像をアップロード","カメラで撮影"))
if img_source == "画像をアップロード":
  img_file =st.sidebar.file_uploader("画像を選択してください。",type=["png","jpg","jpeg"])
elif img_source == "カメラで撮影":
  img_file = st.camera_input("カメラで撮影")

if img_file is not None:
  with st.spinner("判定中..."):
    img = Image.open(img_file)
    st.image(img, caption="対象の画像",width=480)
    st.write("")

    #予測
    results = predict(img)

    #結果の表示
    st.subheader("判定結果")
    n_top = 3 #確率が高い順に３位まで返す
    for result in results[:n_top]:
      st.write(str(round(result[2]*100,2))+"%の確率で"+result[0]+"です。")

    #円のグラフ
    pie_labels = [result[1] for result in results[:n_top]]
    pie_labels.append("others")#その他
    pie_probs=[result[2] for result in results[:n_top]]
    pie_probs.append(sum([result[2]for result in results[n_top:]]))#その他
    fig, ax = plt.subplots()
    wedgeprops = {"width":0.3,"edgecolor":"white"}
    textprops = {"fontsize":6}
    ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90, textprops=textprops, autopct="%.2f",
    wedgeprops=wedgeprops)#円グラフ
    st.pyplot(fig)

  st.sidebar.write("")
  st.sidebar.write("")

  st.sidebar.caption("""
  このアプリは、「Fashion-MNIST」を訓練データとして使っています。\n 
  Copyright(c) 2017 Zalando SE\n
  Released under the MIT license\n
  https://github.com/zalandoreserch/fashion-mnist#license""")
  

