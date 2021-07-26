import csv
from textwrap import *
import pickle
import os.path

csv_path = os.path.abspath(os.path.dirname(__file__)) + "/labels/"

#For printing stuff in color
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#For holding dialogue turns
class Turn:
	def __init__(self, row):
		self.line = row['line']
		self.rid = row['id']
		self.link = row['link'] if row['link'] else None
		self.relationship = row['relationship'] if row['relationship'] else "START"
		self.text = row['text']
		self.speaker = row['speaker']
		self.ami_label = row['ami_label']
		self.topic = row['topic'] if row['topic'] else "NONE"
		self.api_labels = row['api_labels'] if row['api_labels'] else "NONE"

class Node:
	def __init__(self, turn=None):
		self.turns=[]
		if turn: self.turns.append(turn)
		self.children=[]
		self.status = ""

	def __str__(self):
		starter = bcolors.OKGREEN if self.status=="q" else (bcolors.WARNING if self.status=='a' else bcolors.BOLD)
		if not len(self.turns): return ""
		return(" ".join([turn.text for turn in self.turns]))
		# return(starter + self.turns[0].speaker + "("+self.turns[0].relationship + "): " + " ".join([turn.text for turn in self.turns]) + bcolors.ENDC)

	def api_labels(self):
		return list(set([label for turn in self.turns for label in turn.api_labels.split() ]))
	
	def ami_labels(self):
		return list(set([turn.ami_label for turn in self.turns]))

	def topics(self):
		return list(set([turn.topic for turn in self.turns]))

	def speaker(self):
		return self.turns[0].speaker if self.turns else None

	def lines(self):
		return [int(turn.line) for turn in self.turns] if self.turns else None
		# return self.turns[0].line if self.turns else None

	def rid(self):
		return [int(turn.rid) for turn in self.turns] if self.turns else None
		# return self.turns[0].rid

	def relationship(self):
		if self.turns:
			return [self.turns[0].relationship] if self.turns[0].relationship else ["START"]
		else: return None

	def addChild(self, turn):
		self.children.append(Node(turn))

	def addTurnToSelf(self, turn):
		self.turns.append(turn)

	def addTurn(self, turn):
		if turn.link in [t.rid for t in self.turns] or not turn.link:
			if turn.relationship=="CONT":
				self.addTurnToSelf(turn)
				return True
			else:
				self.addChild(turn)
				return True
		else:
			for child in self.children:
				if child.addTurn(turn): return True
		return False

#Extract row from csv
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

#Extract dialogue threads from a conversation tree
def getSequences(node, func, seq=[], level=0):
	sequences = []
	currentSequence = seq + [func(node)] if func(node) else seq
	# if not node.children:
	# 	sequences.append(currentSequence)
	for child in node.children:
		if not node.relationship(): 
			sequences += [getSequences(child, func, currentSequence, level+1)]
		else: 
			currentSequence = getSequences(child, func, currentSequence, level+1)
	if not node.relationship(): return sequences
	else: return currentSequence

#Print dialogue threads
def printTree(node, level=0):
	for line in wrap(str(node)):
		scolor = ""
		if "Apiza" in node.speaker(): scolor = bcolors.OKBLUE
		else: scolor = bcolors.WARNING
		# if "confirm" in node.ami_labels(): 
		print("\t"*level+scolor+node.speaker()+bcolors.ENDC+": " + line )
	print()
	for child in node.children:
		printTree(child, level+1)

#Search each dialogue for some term
def searchAllSequences(sequences, search):
	results = []
	for _ in list(range(1,31)):
		for sequence in all_sequences[_]:
			if search in " ".join(sequence): results.append(sequence)
	return results

if __name__ == "__main__":
	all_dialogues = {}
	all_sequences = {}
	#Parsing each file and storing data in the above dicts
	for _ in list(range(10,18)):
		tree = Node()
		library = "allegro" if _ >14 and _ != 29 else "libssh"
		dialogue = {"lib": library, "utterances": {}}
		print(_)
		with open(csv_path+ "apiza_labels_{:02d}.csv".format(_), newline='') as csvfile:
			reader = csv.DictReader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			for row in reader:
				dialogue['utterances'][int(row['id'])] = make_message(row)
				tree.addTurn(Turn(row))
		all_dialogues[_] = dialogue
		printTree(tree)
		sequences = getSequences(tree, Node.rid)
		all_sequences[_] = sequences

	# pickle.dump(all_sequences, open('sequences.pkl', 'wb'))
