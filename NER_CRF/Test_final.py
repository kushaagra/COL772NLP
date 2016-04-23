import os
import json
import re
import codecs
import sys



input_file = sys.argv[1]
output_file = sys.argv[2]


measurement_units = ['ft','sqft','meter','mt','sq','feet','meters','mts']

def contains_url(input):
	res = re.findall('https?://[^\s]*',input);
	if(len(res)>0):
		return True
	else:
		return False


def convert_name(string_input):
	s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', string_input)
	return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)



def contains_hashtag(input):
	res = re.findall('#',input)
	if(len(res) > 0):
		return True
	else:
		return False

def contains_atrate(input):
	res = re.findall('@',input)
	if(len(res) > 0):
		return True
	else:
		return False

def contains_p_c(input):
	res = re.findall(r'[0-9]+/-',input)
	if(len(res) > 0):
		return True
	else:
		if (input=="lac" or input=="lacs" or input=="crore" or input=="cr"):
			return True
		else:
			return False

def contains_bhk(input):
	res = re.findall('bhk',input)
	if(len(res) > 0):
		return True
	else:
		res = re.findall('kothi',input)
		if(len(res) > 0):
			return True
		return False

def contains_floor(input):
	res = re.findall('floor',input)
	if(len(res) > 0):
		return True
	else:
		return False

def contains_measurement_units(input):
	for item in measurement_units:
		if(item == input):
			return True
		
	return False

def read_brown():
	f = open('paths 2','r')
	brown_cluster = {}
	for line in f:
		print line
		lines = line.split('\t')
		brown_cluster[lines[1]] = lines[0]
	return brown_cluster 


def execute_command_line(command):
	os.system(command)

def read_file(filename,input_data):
	f = open(filename,'r')
	for line in f:
		line = line.rstrip('\n')
		line = line.rstrip('\r')
		#temp = line[-1]
		#line = line[:-1]
		line = line.rstrip();
		#line = line+temp;
		input_data.append(line)
	f.close()


def getSubstring(input_string):
	list = []
	for i in range(len(input_string)):
		for j in range(i+1,len(input_string)):
			if((j-i) > 2):
				list.append(input_string[i:j+1])
	return list



def isAnumber(input):
	if input.isdigit():
		return 1
	else :
		return 0


def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def isCapital(input):
	
	if(input[0].isupper()):
		return 1
	else:
		return 0

def get_names():
	f = open('names.txt')
	names = {}
	counter = 1
	for line in f:
		line = line.rstrip('\n')
		line = line.rstrip('\r')
		names[line.lower()] = counter

	f.close()


	f = open('surnames.txt')
	
	for line in f:
		line = line.rstrip('\n')
		line = line.rstrip('\r')
		names[line.lower()] = counter
		
	f.close()
	return names

def is_name_present(name_list,name):
	try:
		name_list[name]
		return True
	except KeyError:
		return False

def is_location_present(location_list,name):
	try:
		location_list[name]
		return True
	except KeyError:
		res = re.findall("sector",name)
		if(len(res) > 0):
			return True
		return False

def get_locations():
	f = open('NCR_Locations.txt')
	locations = {}
	counter = 1
	for line in f:
		line = line.rstrip('\n')
		line = line.rstrip('\r')
		lines = line.split(" ",1)
		for l in lines:
			locations[l.lower()] = counter

	f.close()
	return locations


def number_of_numerals(input):
	return sum(c.isdigit() for c in input)



def modify_input_data(input_data):
	mod_input_data = []
	name_list = get_names()
	brown_cluster = read_brown()
	location_list = get_locations()
	for i in range(len(input_data)):
		line = input_data[i]
		#print line
		if(len(line)>0):

			lines = line.split(' ')


			hashtag = ""
			if(contains_hashtag(lines[0])):
				hashtag = "HASHTAGYES"
			if(len(lines[0]) > 1):
				lines[0] = re.sub('#','',lines[0],1)
			
			atrate = ""
			if(contains_atrate(lines[0])):
				atrate = "ATRATEYES"
			if(len(lines[0]) > 1):
				lines[0] = re.sub('@','',lines[0],1)
			#print "Line is = " + lines[0]
			url_contain = ""
			if(contains_url(lines[0])):
				url_contain = "URLYES"
				lines[0] = "URL";

			p_c = ""
			if(contains_p_c(lines[0])):
				p_c = "YESPRPRELATED"
			
			
			
			#final_line = line
			number_line =""
			capital_line = ""
			substring_line = ""


			similar_word = ""
			name_list_presence = ""
			location_list_presence = ""

			brown_cl = ""

			prev_word = ""
			if(i >0):
				l = input_data[i-1]
				ls = l.split(' ')
				previous_word = ls[0]
				if(not(previous_word == '\n')):
					prev_word = prev_word + "PREVWORD="+previous_word 

			
			next_word = ""
			if(i<(len(input_data)-1)):
				l = input_data[i+1]
				ls = l.split(' ')
				next_word = ls[0]
				if(not(next_word == '\n')):
					next_word = "NEXTWORD="+next_word


			camel_case_feature = convert_name(lines[0])
			if(camel_case_feature == lines[0].lower()):
				camel_case_feature = ""




			number_numerals = "NUMERAL" + str(number_of_numerals(lines[0]))
			floor_container = ""
			if(contains_floor(lines[0])):
				floor_container = floor_container + "FLOORYES"
			bhk_container = ""
			if(contains_bhk(lines[0])):
				bhk_container = bhk_container + "BHKYES"
			measurement_units_container = ""
			if(contains_measurement_units((lines[0]).lower())):
				measurement_units_container = "MEASYES"

			try:
				brown_cluster[lines[0]]
				for temp in range(5,len(brown_cluster[lines[0]])):
					brown_cl = brown_cl +  " " + brown_cluster[lines[0]][0:temp]
				#brown_cl = brown_cl + " " + brown_cluster[lines[0]][:-2]
				#print "Present brown cluster"
			except KeyError:
				print "Not present brown cluster"


			number = isAnumber(lines[0])
			Capital  = isCapital(lines[0])
			if(number == 1):
				number_line = number_line + "NUMBER"
			if(Capital == 1):
				capital_line = capital_line + "CAPITALIZED " + lines[0].lower()

			listi = getSubstring(lines[0])
			for i in range(len(listi)):
			 	substring_line = substring_line + " " + listi[i]

			if(is_name_present(name_list,(lines[0]).lower())):
				name_list_presence = name_list_presence + "NAMEPRESENT"
				#print "yohooo " + lines[0]

			if(is_location_present(location_list,(lines[0]).lower())):
				location_list_presence = location_list_presence + "LOCATIONPRESENT"
				#print "LOCPRESENT"


			featuress = number_line + " " + capital_line + " " + name_list_presence + " " + similar_word + " " + brown_cl + " " + number_numerals + " "+ location_list_presence+ " " +prev_word+ " "
			featuress = featuress + " " + floor_container + " " + bhk_container + " " + measurement_units_container+ " " + next_word + " "
			featuress = featuress + " " + atrate + " " + hashtag + " " + url_contain + " " + p_c + " "
			final_line = (lines[0]).lower() + " " + substring_line + " " +featuress +" " +camel_case_feature  + "\n"
			final_line = re.sub(" +"," ",final_line)
			
			

				
		else:
			#print "Here where I shouldn;t be"
			final_line = line + "\n"

		mod_input_data.append(final_line)
	return mod_input_data



input_data = []
read_file(input_file,input_data)
mod_input_data = modify_input_data(input_data)
f1 = open(output_file,'w')
for line in mod_input_data:
	f1.write(line)
f1.close()
#execute_command_line('java -cp class:lib/mallet-deps.jar cc.mallet.fst.SimpleTagger --model-file crf_trained --include-input false testing_data_gen > myout.txt') 

