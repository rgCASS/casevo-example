# Casevo Example

There are examples of [Casevo](https://github.com/rgCASS/casevo).

## 1. Description of the Simulation Scenario

The goal is to construct a beginner-level social simulation experiment. In the U.S. presidential elections, candidates organize debates, which are broadcast on television, allowing candidates to express their views and attract votes. However, the effects of debates are not easy to quantify or measure. To address this issue, we can create a virtual pool of voters to simulate the process of voters watching debate content, discussing with each other, and ultimately voting, thereby evaluating the effectiveness of the candidates' debates. This design not only ensures that agents make organized decisions but also reflects the influence of information exchange and individual experiences on final decisions through memory mechanisms and reflection processes.

Key information in the simulation experiment includes:

- Debate Content: Uses transcripts of the 2020 U.S. presidential television debates [Link1](https://www.debates.org/voter-education/debate-transcripts/october-22-2020-debate-transcript/) [Link2](https://www.presidency.ucsb.edu/documents/presidential-debate-belmont-university-nashville-tennessee-0)
  - Divided into 6 parts placed in the `content` folder.
- Voter Profiles: Derived from [political voter research article](https://www.pewresearch.org/politics/2021/11/09/beyond-red-vs-blue-the-political-typology-2/)
  - A total of 9 voter profiles, with 3 selected and configured in JSON format in the `person.json` file.
- Network Structure:
  - A random network with 9 nodes, using a complete network.

The overall simulation process can be described as follows:

- The simulation is divided into 6 rounds, with each round corresponding to a portion of the debate content, including events such as:
  - Public Debate: Agents receive information about presidential candidates' debates during this phase, mainly understanding and recording candidates' statements and policy positions. Each agent makes initial judgments based on their views and preferences.
  - Talk: After the debate, agents discuss with other voters. This is a critical stage for information exchange and conflict of views among agents. Discussion topics can include opinions on the debate, discussions on candidates' policies, or sharing of individual experiences.
  - Reflect: After the discussions, each agent engages in self-reflection. During this phase, agents conduct comprehensive thinking based on the content of the discussions, their understanding of the debate, and information in their memory. The reflection process introduces a memory mechanism, allowing agents to adjust their views on candidates based on past experiences and memories.
  - Vote: Each agent makes a voting decision based on the outcomes of the debate, discussion, and reflection. This step is the final output of the entire process, where all agents' decisions are reflected in the voting results.


## 2. Directories and Files

- `election_example` project folder.
  - `content`: Texts of the debate content, named by rounds as `1.txt, 2.txt, ...`
  - `prompt`: Templates for all prompts
  - `log`: Directory for log output
  - `memory`: Directory for memory vector database
  - `build_case.py`: Simulation configuration file
  - `election_model.py`: Simulation scenario file
  - `election_agent.py`: Agent file
  - `llms`: LLM Interface for Simulations
    - `baichuan.py`: Large model interface file, implementing the Baichuan API in this example
    - `ollama_lib.py`: Interface for ollama
  - `run.py`: Simulation entry point

## 3. Simulation Execution

Run the simulation by executing the following command in the command line:

```cmd
python run.py case_lite.json 6
```

Where `case_lite.json` is the configuration file, and `6` is the number of simulation rounds.

PS: In the example, you need to replace `API_KEY` in `run.py` with a valid Baichuan large model API_KEY.