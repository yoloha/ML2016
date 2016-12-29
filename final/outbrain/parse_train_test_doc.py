fw = open('clicks_train_doc.csv', 'wb')
f_train = open('clicks_train.csv', 'r')
f_doc = open('events_document.csv','r')
# Skip header
f_train.readline()
f_doc.readline()
display_id = -1
fw.write("display_id,ad_id,document_id,clicked\n")
while True:
	line_train = f_train.readline()
	if not line_train: break
	line_train = line_train.rstrip('\n')
	ls_train = line_train.split(',')
	if ls_train[0] != display_id:
		line_doc = f_doc.readline()
		line_doc = line_doc.rstrip('\n')
		ls_doc = line_doc.split(',')
		display_id = ls_train[0]

	ls_p = [ls_train[0], ls_train[1], ls_doc[1], ls_train[2]]
	fw.write(",".join(map(str,ls_p)))
	fw.write("\n")

fw = open('clicks_test_doc.csv', 'wb')
f_test = open('clicks_test.csv', 'r')
# Skip header
f_test.readline()
fw.write("display_id,ad_id,document_id\n")
while True:
	line_test = f_test.readline()
	if not line_test: break
	line_test = line_test.rstrip('\n')
	ls_test = line_test.split(',')
	if ls_test[0] != display_id:
		line_doc = f_doc.readline()
		line_doc = line_doc.rstrip('\n')
		ls_doc = line_doc.split(',')
		display_id = ls_test[0]

	ls_p = [ls_test[0], ls_test[1], ls_doc[1]]
	fw.write(",".join(map(str,ls_p)))
	fw.write("\n")