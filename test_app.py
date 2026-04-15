import streamlit as st
import os

st.title("Test App - Debug Mode")

# Show current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
st.write(f"📁 Current directory: {current_dir}")

# List all files in directory
if os.path.exists(current_dir):
    files = os.listdir(current_dir)
    st.write("📄 All files in directory:")
    for f in files:
        st.write(f"  - {f}")
    
    # Check for .pkl files
    pkl_files = [f for f in files if f.endswith('.pkl')]
    st.write(f"\n🔍 Found {len(pkl_files)} .pkl files:")
    for f in pkl_files:
        st.write(f"  ✅ {f}")

st.success("If you see this, the app is working!")
