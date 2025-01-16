from election_agent import ElectionAgent

from casevo import ModelBase

from casevo import TotLog


#测试样例模型
class ElectionModel(ModelBase):

    #根据config生成model
    def __init__(self, tar_graph, person_list, llm):
        """
        初始化对话系统中的每个人物和他们的对话流程。

        :param tar_graph: 目标图，表示对话系统的结构。
        :param person_list: 人物列表，包含所有参与对话的人物信息。
        :param llm: 语言模型，用于生成对话内容。
        """
        
        super().__init__(tar_graph, llm)
        
        #设置Agent        
        for cur_id in range(len(person_list)):
            cur_person = person_list[cur_id]
            cur_agent = ElectionAgent(cur_id, self, cur_person, None)
            self.add_agent(cur_agent, cur_id)
            
            
    def public_debate(self):
        """
        所有Agent听取一阶段的公开辩论。

        此函数模拟了一次公开辩论的过程。它首先打印出辩论开始的时间，然后读取当前辩论的主题总结，
        接着让每个参与者（agent）听取辩论总结。最后，它将此次辩论的内容记录到日志中，并打印出辩论结束的时间。
        
        该方法不接受任何参数，也没有返回值。
        """
        
        print('public_debate: %d start' % self.schedule.time)
        TotLog.add_model_log(self.schedule.time, 'public_debate', log_item)
        #获取辩论文本
        cur_debate_num = self.schedule.time + 1
        with open('content/%d.txt' % cur_debate_num) as f:
            cur_debate_summary = f.read()
        
        #循环每个agent听取辩论文本
        for cur_agent in self.agents:
            cur_agent.listen(cur_debate_summary, 'public')
        
        #事件加入日志
        log_item = {
            'debate_content': cur_debate_summary
        }

        print('public_debate: %d end' % self.schedule.time)   

    def reflect(self):
        """
        所有Agent进行一阶段的反思。
        """
        for cur_agent in self.agents:
            cur_agent.reflect()
    

    #整体模型的step函数
    def step(self):
        #听取辩论文本
        self.public_debate()
        #节点自由讨论
        self.schedule.step()
        
        #节点反思
        self.reflect()
        return 0
  
       