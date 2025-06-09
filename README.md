Hello people

This is a test github repository of our Daejeon Apartment Listing for Real Estate Agents Assistant that will be helpful for International Students (mainly in Woosong University, but it also could be used for Daejeon as a whole).

Some things to note for this code version:
- You may need to do these code steps for initializing the virtual environment, since this uses venv:

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

- You also need to have Ollama installed in your system, since this needs llama3 for its LLM and RAG features

- For running the code, you can type this in command prompt (or any equivalent):
  streamlit run app.py

- Although the dataset used doesn't include images, as it is mainly used because it is available for free and public use (while some dynamic datasets are not easily accesible), this code is able to display images if the dataset include them, so it 'should' be able to output images too.

- Will be trying to deploy a version that is publicly online in a server, though it may exclude certain functionalities because of compatibility/efficiency reasons.
  Update on deployment: was not able to due to venv making it virtualized envrionment that is helpful for local systems, will try maybe to see if this can be solved
