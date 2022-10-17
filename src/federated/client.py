from federatedscope.core.communication import gRPCCommManager
from federatedscope.core.message import Message

class Client(object):
    def __init__(self, client_num, client_id, host_port, **kwargs):
        self.client_num = client_num
        self.client_id = client_id + 1
        self.host_port = str(host_port)
        self.client_port = str(int(self.host_port) + self.client_id)
        self.manager = gRPCCommManager(
            host = "127.0.0.1", 
            port = self.client_port, 
            client_num = self.client_num)
        self.manager.add_neighbors(0, address = {"host": "127.0.0.1", "port": self.host_port})

    def join(self):
        self.manager.send(Message(
            msg_type = "join", 
            sender = self.client_id, 
            receiver = [0], 
            content = f"127.0.0.1:{self.client_port}"))
        msg = self.manager.receive()
        assert msg.msg_type == "start", "Should get start message here"

    def communicate_weight(self, content):
        self.manager.send(Message(
            msg_type = "client_weight", 
            sender = self.client_id, 
            receiver = [0], 
            content = content))
        msg = self.manager.receive()
        assert msg.msg_type == "host_weight", "Should get host weight here"
        return msg.content

    def end(self):
        self.manager.send(Message(
            msg_type = "end", 
            sender = self.client_id, 
            receiver = [0], 
            content = "end"))

        
