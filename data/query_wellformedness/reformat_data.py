import sys
import os

if __name__=='__main__':
	if len(sys.argv) < 2:
		print("Expected filename as an argument")
		sys.exit()
	filepath = sys.argv[1]
	path, filename = os.path.split(filepath)
	name, ext = os.path.splitext(os.path.basename(filename))
	new_filepath = os.path.join(path, 'processed_'+name+'.txt')
	with open(new_filepath, 'w') as new_file:
		with open(filepath, 'r') as old_file:
			for line in old_file:
				question, number = line.strip().split('\t')
				y = '2' if float(number) >= 0.5 else '1'
				label = '__label__'+y
				new_line = label + ' , ' + question + '\n'
				new_file.write(new_line)
	print('Finished')
