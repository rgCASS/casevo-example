from casevo import AgentBase
from casevo import BaseStep, JsonStep
from casevo import TotLog

#定义CoT中的步骤
class OpinionStep(BaseStep):
    def pre_process(self, input, agent=None, model=None):
        cur_input = input['input']
        cur_input['issue'] = input['last_response']
        return cur_input

class AgreeStep(JsonStep):
    def pre_process(self, input, agent=None, model=None):
        cur_input = input['input']
        cur_input['opinion'] = input['last_response']
        return cur_input


#样例agent
class ElectionAgent(AgentBase):
    
    def __init__(self, unique_id, model, description, context):
        #初始化
        super().__init__(unique_id, model, description, context)

        #加载Prompt
        issue_prompt = self.model.prompt_factory.get_template("issue.txt")
        opinion_prompt = self.model.prompt_factory.get_template("opinion.txt")
        agree_prompt = self.model.prompt_factory.get_template("agree.txt")
        talk_prompt = self.model.prompt_factory.get_template("talk.txt")

        #设置CoT
        issue_step = BaseStep(0, issue_prompt)
        opinion_step = OpinionStep(1, opinion_prompt)
        agree_step = AgreeStep(2, agree_prompt)
        talk_step = BaseStep(3, talk_prompt)
        
        chain_dict = {
            'listen': [issue_step, opinion_step, agree_step],
            'talk': [talk_step]
        }
        self.setup_chain(chain_dict)
        
        
    #以下定义Agent的相关行为函数
    
    def listen(self, content, source):
        """
        用于处理Agent获得的输入内容（听到的内容）并根据来源存储相关信息。

        参数:
        content: 字符串，表示需要处理的输入内容。
        source: 字符串，表示输入内容的来源。

        返回值:
        无直接返回值，但会通过内存和日志模块记录处理结果。
        """
        
        #构建CoT的输入
        input_item = {
            'content': content,
            'source': source
        }
        #设置CoT的输入
        self.chains['listen'].set_input(input_item)
        #运行CoT
        self.chains['listen'].run_step()
        
        #处理输出
        tmp_result = self.chains['listen'].get_output()['input']
        tmp_result['agree'] = self.chains['listen'].get_output()['json']
        if source == 'public':
            self.public_listen_result = tmp_result

        #加入记忆模块
        self.memory.add_short_memory(source, self.component_id, 'opinion', tmp_result['opinion'])
        self.memory.add_short_memory(source, self.component_id, 'agree', tmp_result['agree']['特朗普']['总体看法'])
        self.memory.add_short_memory(source, self.component_id, 'agree', tmp_result['agree']['拜登']['总体看法'])
       
        
        #记录日志
        log_item = {
            'source': source,
            'opinion': tmp_result['opinion'],
            'agree': tmp_result['agree']
        }
        TotLog.add_agent_log(self.model.schedule.time,'listen', log_item, self.unique_id)

        
    def talk(self, tar_agent):
        """
        生成与目标代理进行对话行为。
        
        本函数实现了本代理与目标代理之间的信息交流。它首先根据本代理的公开监听结果和长期记忆
        构建输入项，然后通过对话链进行处理。处理完成后，提取最后一次响应作为对话内容，并记录
        对话日志。最后，将对话内容传递给目标代理进行监听。
        
        :param tar_agent: 目标代理对象，即与本代理进行对话的代理。
        """
        #设置CoT的输入
        input_item = {
            'opinion': self.public_listen_result['opinion'],
            'long_memory': self.memory.get_long_memory()
        }
        
        self.chains['talk'].set_input(input_item)
        #运行CoT
        self.chains['talk'].run_step()
        #处理输出
        self.talk_content = self.chains['talk'].get_output()['last_response']
        
        #记录log
        log_item = {
            'target': tar_agent.component_id,
            'content': self.talk_content
        }
        TotLog.add_agent_log(self.model.schedule.time,'talk', log_item, self.unique_id)

        #目标Agent听取内容
        tar_agent.listen(self.talk_content, self.component_id)
    
    def reflect(self):
        """
        执行反思过程。

        该方法在 agent 的执行周期结束后调用，用于反思其记忆中的长期意见。
        它通过比较反思前后的意见变化来记录和学习。

        参数:
        无

        返回值:
        无
        """
        ori_opinion = self.memory.get_long_memory()
        #执行反思
        self.memory.reflect_memory()
        
        #记录Log
        log_item = {
            'ori_opinion': ori_opinion,
            'new_opinion': self.memory.get_long_memory()
        }
        TotLog.add_agent_log(self.model.schedule.time,'reflect', log_item, self.unique_id)

    def step(self):
        """
        执行Agent的一步操作。

        这个方法描述了Agent如何在每一步中进行决策。具体来说，它以一定的概率选择与邻居中的另一个Agent进行交流。
        """
        # 以40%的概率决定是否进行交流动作
        if self.random.random() < 0.4:     
            #随机获取邻居
            neighbors = self.model.grid.get_neighbors(self.unique_id, include_center=False)
            other_agent = self.random.choice(neighbors)
            
            #Agent之间进行对话
            self.talk(other_agent)     


 