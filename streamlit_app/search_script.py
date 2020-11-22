import streamlit as st
import pandas as pd
import numpy as np

from index import ExampleIndex

st.title('Player Search')

#Set up
embeddings_file_path = "dnn_embeddings_sh_v2.csv"
embeddings = pd.read_csv(embeddings_file_path)
embeddings_columns = [f'PC_{i}' for i in range(0,32)]


def get_dataset(df, embeddings_columns):
    vectors = np.ascontiguousarray(df[embeddings_columns])
    labels = np.ascontiguousarray(df['player_fifa_api_id'])

    return vectors.astype('float32'), np.expand_dims(labels, axis=1)

# Define Search Functions
class Service_Index():
    def __init__(self, df, embeddings_columns):
        vectors, labels = get_dataset(df, embeddings_columns)
        self.df = df
        self.vectors = vectors
        self.labels = labels

        self.index = ExampleIndex(vectors, labels)
        self.index.build()

        self.columns = ['player_name', 'overall_rating', 'potential', 'player_positions','birthday',
                        'player_fifa_api_id']

    def search_by_player_name(self, player_name):
        # player_name must be exact
        query_id = self.df[self.df.player_name == player_name].player_fifa_api_id.values[0]
        id_ = self.df[self.df.player_fifa_api_id == query_id].index.values[0]
        results, distances = self.index.query(np.expand_dims(self.vectors[id_], axis=0), k=11)
        response = []
        for ind, result_id in enumerate(results):
            if query_id != result_id:
                result_dict = self.df[self.df.player_fifa_api_id == result_id[0]][self.columns].iloc[0].to_dict()
                result_dict["similarity_score"] = 1/(1+distances[ind])
                response.append(result_dict)
        return response

    def search_by_vector(self, vector, query_id):
        results, distance = self.index.query(np.expand_dims(vector, axis=0), k=11)
        response = []
        for result_id in results:
          if result_id != query_id:
            response.append(self.df[self.df.player_fifa_api_id == result_id[0]][self.columns].iloc[0].to_dict())
        return response


# returns DF from list of dicts
def getDf(search_result, scores_only = False):
  def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

  result_df = pd.DataFrame()
  
  for result in search_result:
    result_df = result_df.append(result, ignore_index=True)
  if scores_only:
    result_df = result_df[['player_name', 'similarity_score','overall_rating', 'potential']]
    result_df = result_df.sort_values('similarity_score', ascending = False)
    columns_to_highlight = ['overall_rating', 'potential', 'similarity_score']
  else:
    if 'similarity_score' in result_df.columns:
      result_df = result_df[['player_name', 'overall_rating', 'potential', 'similarity_score', 'player_positions','birthday','player_fifa_api_id']]
    else:
      result_df = result_df[['player_name', 'overall_rating', 'potential', 'player_positions','birthday','player_fifa_api_id']]
    # result_df = result_df.sort_values('similarity_score', ascending = False)
    columns_to_highlight = ['overall_rating', 'potential']
  return result_df.style.apply(highlight_max, axis=0, subset=pd.IndexSlice[:,columns_to_highlight]) # 

def stSearchDb():
  st.sidebar.header("Use Case 1: Search all in DB")
  player_name = st.sidebar.text_input("Type player name", key = "1")

  if player_name:
    all_index = Service_Index(embeddings, embeddings_columns)
    search_result = all_index.search_by_player_name(player_name)
    st.dataframe(data=getDf(search_result))
    return

def stFilterSearch():
  embeddings.birthday = pd.to_datetime(embeddings.birthday)

  st.sidebar.header("Use Case 2: Filter-Search")
  # date_to_query = st.sidebar.text_input("Type earliest DOB in this format: YYYYMMDD (eg. 19950101)", key="2a")
  # year_input = st.sidebar.slider('Select earliest DOB', 1975, 2010, 1995, format = 'Birth Year %d')
  # date_to_query = str(year_input) + "0101"
  max_age = st.sidebar.slider('Select max age of related players', 10, 45, 21, format = '%d y.o.')
  year = 2016 - max_age
  date_to_query = str(year) + "0101"
  player_name = st.sidebar.text_input("Type player name", key="2b")

  if date_to_query and player_name:
    new_df = embeddings.query('birthday > ' + date_to_query)

    query_id = embeddings[embeddings.player_name == player_name].player_fifa_api_id.values[0]
    query_vector = np.array(embeddings[embeddings.player_fifa_api_id == query_id][embeddings_columns])
    query_vector = query_vector.reshape((32)).astype('float32')

    result_df = pd.DataFrame()
    index = Service_Index(new_df, embeddings_columns)
    response = index.search_by_vector(query_vector, query_id)
    st.dataframe(data=getDf(response))
    return

def stBestSearch():
  st.sidebar.header("Use Case 3: Search and compare similarity metrics")
  player_name = st.sidebar.text_input("Type player name", key="3")
  if player_name:
    all_index = Service_Index(embeddings, embeddings_columns)
    response = all_index.search_by_player_name(player_name)
    sorted_df = getDf(response, True)
    st.dataframe(data=sorted_df)
    return

def main():
  st.sidebar.title("Choose Options")
  st.subheader("Once an option is selected, a table of closely-related players will be generated. Important values are highlighted in yellow.")
  
  # st.header("USE CASE 1: Search all in DB")
  stSearchDb()

  # st.header("Use Case 2: Filter-Search")
  stFilterSearch()

  # st.header("Use Case 3: Best-Search")
  stBestSearch()



if __name__ == "__main__":
  main()
