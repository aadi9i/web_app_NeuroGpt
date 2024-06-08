from muselsl import *

# muses = list_muses(backend="bluemuse")
# print(muses)
def callback(data,timestamps):
    print(data)

stream(address=None,backend="bluemuse")
view()
# streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)
# inlet = StreamInlet(stream, max_chunklen=LSL_EEG_CHUNK)
# info = inlet.info()
# Note: Streaming is synchronous, so code here will not execute until after the stream has been closed
print('Stream has ended')