import csv
from textwrap import *
import pickle

def make_message(row):
	line = row['line']
	rid = row['id']
	link = row['link'] if row['link'] else None
	relationship = row['relationship'] if row['relationship'] else None
	text = row['text']
	speaker = row['speaker']
	ami_label = row['ami_label']
	topic = row['topic'].split() if row['topic'] else None
	api_labels = row['api_labels'].split() if row['api_labels'] else None
	return {'message_id': line, 'text': text, 'speaker': speaker, 'link': link, 'backward-facing': relationship, "illocutionary": ami_label, "api": api_labels, "traceability": topic}

# count = 0
all_dialogues = {}
for _ in list(range(10,18)):
	# tree = Node()
	library = "allegro" if _ >14 and _ != 29 else "libssh"
	dialogue = {"lib": library, "utterances": {}}
	print(_)
	with open('labels/apiza_labels_{:02d}.csv'.format(_), newline='') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for row in reader:
			dialogue['utterances'][int(row['id'])] = make_message(row)
	all_dialogues[_] = dialogue


pickle.dump(all_dialogues, open('annotations.pkl', 'wb'))

