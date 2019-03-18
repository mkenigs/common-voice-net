# Project Structure
* normalize.sql:
  * Normalizes common voice data set and loads into sqlserver

* frontend:
  * Input: mp3, list of sentences from common voice data set
  * Output: classification of mp3
  * Uses JS or PHP to SELECT subset from SQL database. Sends subset to neural net. Waits for neural net to train. Sends neural net mp3 and list of sentences, gets classification back

* Neural net:
  * Input 1: subset
  * Output 1: nothing, but trains on subset
  * Input 2: mp3 and list of sentences
  * Output 2: classification
