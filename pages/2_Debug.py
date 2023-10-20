import streamlit as st
import sys

st.set_page_config(
    layout="wide",
    page_title="Debug",
)

st.write("# Debug Page")


def deep_getsizeof(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    # Mark this object as seen
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([deep_getsizeof(v, seen) for v in obj.values()])
        size += sum([deep_getsizeof(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += deep_getsizeof(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([deep_getsizeof(i, seen) for i in obj])
        
    return size

st.session_state

for device_name, device in st.session_state.devices_list.items():
    with st.expander(device_name, expanded=False):
        col1, col2 = st.columns(2)
        col1.write(device['data'])
        col2.write(device['settings'])
    
size = deep_getsizeof(st.session_state.devices_list) / (1024 * 1024)
# print size in megabytes
st.write("Size of the data: ", round(size, 2), " MB")    
        