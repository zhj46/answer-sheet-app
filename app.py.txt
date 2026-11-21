import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime
import os
from supabase import create_client, Client
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import requests

# --- 1. 初始化与配置 ---
st.set_page_config(page_title="个人答题卡智能阅卷系统", layout="wide")
st.title("📄 个人答题卡智能阅卷系统")

# 从 Streamlit Secrets 读取敏感信息
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    st.error("⚠️ 缺少配置。请点击网页右上角的 'Settings' -> 'Secrets'，填入 SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY。")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# --- 2. 数据库操作函数 ---
def get_exams():
    res = supabase.table("exams").select("*").order("created_at", desc=True).execute()
    return res.data

def save_exam(exam_data):
    res = supabase.table("exams").insert(exam_data).execute()
    return res.data

def upload_image(file_bytes, file_name):
    try:
        supabase.storage.from_("answer_sheets").upload(
            path=file_name,
            file=file_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        return supabase.storage.from_("answer_sheets").get_public_url(file_name)
    except Exception as e:
        st.error(f"图片上传失败: {e}")
        return None

def get_grading_results(exam_id):
    res = supabase.table("grading_results").select("*").eq("exam_id", exam_id).execute()
    return res.data

def save_grading_result(result_data):
    res = supabase.table("grading_results").upsert(result_data).execute()
    return res.data

# --- 3. 图像处理与 OMR (选择题识别模拟) ---
def process_omr_image(image_bytes, answer_key):
    # 注意：这里为了跑通流程，使用了模拟结果。
    # 真正的OMR需要极复杂的 OpenCV 坐标计算，建议后续人工核对选择题。
    mock_results = {q: "A" for q in answer_key.keys()} 
    return mock_results

# --- 4. AI 辅助阅卷 ---
def grade_subjective_with_ai(question_no, student_answer, reference_answer, full_marks, rubric=""):
    prompt = f"""
    你是一个严谨的阅卷老师。请根据信息批改主观题。
    题目: {question_no}, 满分: {full_marks}
    参考答案: {reference_answer}
    学生作答: {student_answer}
    
    请输出JSON格式: {{"suggested_score": 分数(数字), "feedback": "详细点评(优点+不足)"}}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 如果用的是国内中转API，请换成对应的模型名，如 "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.content)
        result['suggested_score'] = min(float(result.get('suggested_score', 0)), float(full_marks))
        return result
    except Exception as e:
        return {"suggested_score": 0, "feedback": f"AI评分异常: {str(e)}"}

# --- 5. PDF 导出 ---
def generate_annotated_pdf(original_image_url, grading_results, total_score, full_score):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    try:
        img_data = requests.get(original_image_url).content
        img_pil = Image.open(io.BytesIO(img_data))
        img_width, img_height = img_pil.size
        aspect = img_height / img_width
        display_width = width - 40
        display_height = display_width * aspect
        if display_height > height - 100:
            display_height = height - 100
            display_width = display_height / aspect
        c.drawImage(ImageReader(img_pil), 20, height - display_height - 20, width=display_width, height=display_height)
    except:
        c.drawString(50, height - 50, "无法加载原图")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 30, f"总分: {total_score} / {full_score}")
    c.setFont("Helvetica", 10)
    y_position = height - display_height - 50 if 'display_height' in locals() else 100
    
    for res in grading_results:
        if y_position < 50:
            c.showPage()
            y_position = height - 50
        line = f"第{res['question_no']}题: {res['score']}分 (满分{res['full_marks']}分) - {res.get('feedback', '')[:30]}..."
        c.drawString(50, y_position, line)
        y_position -= 15

    c.save()
    buffer.seek(0)
    return buffer

# --- 6. Streamlit UI ---
if 'current_exam_id' not in st.session_state:
    st.session_state.current_exam_id = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_image_url' not in st.session_state:
    st.session_state.current_image_url = None

with st.sidebar:
    st.header("📂 试卷管理")
    exams = get_exams()
    exam_options = {f"{e['title']}": e['id'] for e in exams}
    selected_exam_name = st.selectbox("选择已有试卷", options=["-- 新建试卷 --"] + list(exam_options.keys()))
    
    if selected_exam_name != "-- 新建试卷 --":
        st.session_state.current_exam_id = exam_options[selected_exam_name]
        exam_detail = supabase.table("exams").select("config").eq("id", st.session_state.current_exam_id).single().execute()
        if exam_detail.data:
            st.session_state.questions = exam_detail.data['config'].get('questions', [])
    else:
        with st.form("new_exam_form"):
            exam_title = st.text_input("试卷标题", "高一物理期末卷")
            if st.form_submit_button("创建空白试卷"):
                default_questions = [{"no": "1", "type": "选择题", "full": 3, "ref": "A", "rubric": ""}]
                new_exam_data = {"title": exam_title, "config": {"questions": default_questions}}
                created = save_exam(new_exam_data)
                st.session_state.current_exam_id = created[0]['id']
                st.session_state.questions = default_questions
                st.rerun()

if st.session_state.current_exam_id is None:
    st.info("👈 请在左侧创建或选择一份试卷以开始。")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📝 制卡配置", "📸 上传原卷", "🤖 AI 辅助阅卷", "📊 成绩汇总与导出"])

with tab1:
    st.subheader("调整题目信息")
    edited_questions = st.data_editor(
        st.session_state.questions,
        num_rows="dynamic",
        column_config={
            "no": "题号", "type": st.column_config.SelectboxColumn("题型", options=["选择题", "填空题", "解答题"]),
            "full": "满分", "ref": "参考答案", "rubric": "评分细则"
        }
    )
    if st.button("保存题目配置"):
        supabase.table("exams").update({"config": {"questions": edited_questions}}).eq("id", st.session_state.current_exam_id).execute()
        st.success("配置已保存！")

with tab2:
    st.subheader("上传答题卡扫描件")
    uploaded_file = st.file_uploader("选择图片 (请确保已在Supabase创建 public 桶 'answer_sheets')", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption="上传的答题卡", use_column_width=True)
        
        if st.button("上传至云端并保存"):
            file_name = f"exam_{st.session_state.current_exam_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            url = upload_image(image_bytes, file_name)
            if url:
                st.session_state.current_image_url = url
                st.success(f"上传成功！当前试卷关联图片已更新。")
                
        if st.session_state.current_image_url:
            st.info(f"当前关联的原卷图片：{st.session_state.current_image_url}")

with tab3:
    st.subheader("AI 辅助主观题评分")
    subjective_questions = [q for q in st.session_state.questions if q['type'] != '选择题']
    
    if not subjective_questions:
        st.info("当前试卷没有主观题。")
    else:
        q_options = {f"第{q['no']}题": q for q in subjective_questions}
        selected_q_key = st.selectbox("选择题目", options=list(q_options.keys()))
        selected_q = q_options[selected_q_key]
        
        col1, col2 = st.columns(2)
        with col1:
            student_answer = st.text_area("学生作答内容", height=150)
        with col2:
            st.info(f"**参考答案:**\n{selected_q.get('ref', '未填写')}")
        
        if st.button("调用 AI 评分"):
            if not student_answer:
                st.warning("请先输入学生作答内容。")
            else:
                with st.spinner("AI 正在分析..."):
                    ai_result = grade_subjective_with_ai(selected_q['no'], student_answer, selected_q.get('ref', ''), selected_q['full'])
                    
                    final_score = st.number_input("确认或修改得分", min_value=0.0, max_value=float(selected_q['full']), value=float(ai_result.get('suggested_score', 0)), step=0.5)
                    st.write(f"**AI 点评:** {ai_result.get('feedback', '')}")
                    
                    if st.button("确认并保存该题得分"):
                        result_record = {
                            "exam_id": st.session_state.current_exam_id,
                            "question_no": selected_q['no'],
                            "score": final_score,
                            "full_marks": selected_q['full'],
                            "feedback": ai_result.get('feedback', '')
                        }
                        save_grading_result(result_record)
                        st.success(f"第{selected_q['no']}题得分已保存至云端！")

with tab4:
    st.subheader("成绩汇总")
    results = get_grading_results(st.session_state.current_exam_id)
    if not results:
        st.info("还没有任何阅卷记录。")
    else:
        total_score = sum(r['score'] for r in results)
        total_full = sum(q['full'] for q in st.session_state.questions)
        st.metric("总分", f"{total_score:.1f} / {total_full}")
        st.dataframe(results)
        
        st.divider()
        st.subheader("导出留痕 PDF")
        if st.button("生成并下载 PDF"):
            if not st.session_state.current_image_url:
                st.warning("请先在 Tab 2 上传原卷图片。")
            else:
                with st.spinner("正在生成PDF..."):
                    pdf_buffer = generate_annotated_pdf(st.session_state.current_image_url, results, total_score, total_full)
                    st.download_button(
                        label="点击这里下载 PDF",
                        data=pdf_buffer,
                        file_name=f"答题卡批改_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )