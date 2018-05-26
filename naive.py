import collections

reactions = ['Love', 'Haha', 'Sad', 'Wow', 'Angry']

# statuses is a passed in list of scraped statuses, where each status 
# is a struct containing a Counter of reactions denoted as status.reactions
# and the text of the status

def baseline(statuses):
	reactions_aggregate = collections.Counter()
	for status in statuses:
		reactions += status.reactions

	baseline_react = max(reactions_aggregate, key=lambda k:k[1])
	output = {reactions[i]:0 for i in range(5)}
	output[baseline_react[0]] = 1
	return output
