# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk.events import SlotSet
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from .models import *
import json
import os
from .document import *
from .stages import *

def send(d: CollectingDispatcher, obj: Any): d.utter_message(str(obj))
def getSlot_StoryStage(t: Tracker): return t.get_slot('story_stage')
def getUserLatestMEG(t: Tracker): return t.latest_message
def getUserText(t: Tracker): return getUserLatestMEG(t)["text"]
def getUserId(t: Tracker): return t.sender_id
client = createClient()
assert(checkClient(client))




class ActionAskGpt(Action):
    def name(self) -> Text:
        return "action_ActionAskGpt"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:


        for line in callGPTByRAG(getUserId(tracker), userStatus['stage'], getUserText(tracker)).split("\n"):
            dispatcher.utter_message(line)       
        updateDocuments(client, [{"key":getUserId(tracker), "value": userStatus}])
        return []


class ActionGoNext(Action):
    def name(self) -> Text:
        return "action_ActionGoNext"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("請重新說明一次您的問題")
        return []    
