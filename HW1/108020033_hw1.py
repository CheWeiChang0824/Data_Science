import sys

transactions = []
item_table = []
cnt = [0] * 1000
len_of_transactions = 0
check_item = [False] * 1000

def Load_Transactions(file_name):

    with open(file_name, 'r') as file:
        for line in file:
            items = [int(item) for item in line.strip().split(',')]
            transactions.append(items)
       
    len_check_item = len(check_item)
    # To record this round is at which number of transaction
    trans_cnt = 0

    # Build up item_table to record item k is at which transaction
    i = 0
    for i in range(1000):
        item_table.append(set())

    for trans in transactions:
        for item in trans:
            # If check[item] is false, then it means it doesn't exist before in this transaction
            if check_item[item] == False:
                item_table[item].add(trans_cnt)
                cnt[item] += 1
                check_item[item] = True
            else:
                continue
        
        # Update trans_cnt
        trans_cnt += 1

        # Refresh the check_item and transaction_list
        for i in range(len_check_item):
            check_item[i] = False
    return 

def Initiate_Candidate():
    Candidate = list()
    len_of_cnt = len(cnt)
    for i in range(len_of_cnt):
        if cnt[i]:
            Candidate.append([i])
    return Candidate

def Count_support(item_set, iteration):
    # count is for calculating frequency of the item_set
    count = 0.0
    
    if iteration == 1:
        count = cnt[item_set[0]]

    else:
        item1 = item_set[0]
        intersect = item_table[item1]
        len_item_set = len(item_set)

        for i in range(len_item_set):
            item2 = item_set[i]
            intersect = intersect & item_table[item2]

        count = len(intersect)
        
    freq = count / len_of_transactions
    return freq  

def Generate_Candidate(Li, iteration):
    Ci = list()
    len_Li = len(Li)
    for i in range(len_Li):
        for j in range(i + 1, len_Li):
            # compare whether k-1 elements are equal when iteration = k + 1
            if Li[i][:iteration - 2] == Li[j][:iteration - 2]:
                mid = [Li[i][iteration - 2], Li[j][iteration - 2]]
                Ci.append(Li[i][:iteration - 2] + mid)
    return Ci
  

# Return the frequent patterns
def Apriori(min_sup):
    # Part 2.1 : Ci is a two dimension list to store C1
    Ci = Initiate_Candidate()

    Li = list()
    FP = list()
    Freq = list()
    iteration = 1
    
    # Part 2.2: Generate Ci and Li in each iteration
    while Ci:
        # In this for loop, we get the Li
        for item_set in Ci:
            freq = Count_support(item_set, iteration)

            # if the freq of item set is bigger or equal than min_sup, add it
            if freq >= min_sup:
                Li.append(item_set)
                FP.append(item_set)
                Freq.append(freq)
        
        iteration += 1
        Ci_next = Generate_Candidate(Li, iteration)
        Ci = Ci_next
        Li.clear()
        
    return FP, Freq

def Write_Output_File(Freq_Patterns, Freq, File_Name):
    with open(File_Name, 'w') as file:
        len_Freq_Patterns = len(Freq_Patterns)
        for i in range(len_Freq_Patterns):
            len_i_Freq_Patterns = len(Freq_Patterns[i])
            for j in range(len_i_Freq_Patterns):
                if j != len_i_Freq_Patterns - 1:
                    file.write(str(Freq_Patterns[i][j]) + ',')
                else:
                    file.write(str(Freq_Patterns[i][j]) + ':')
            file.write("{:.4f}".format(Freq[i]) + '\n')


if __name__ == "__main__":

    # Check the number of parameters is correct or not
    if len(sys.argv) != 4:
        print("Usage: python script.py min_support input_file output_file")
        sys.exit(1)

    # Take the parameters
    min_support = float(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    # Part 1: Load transactions, transactions is a two dimension list to store the input sorted data
    Load_Transactions(input_file)
    len_of_transactions = len(transactions)

    # Part 2: Do the Apriori algorithm
    Freq_Patterns, Freq = Apriori(min_support)

    # Part 3: Write the result into the file
    Write_Output_File(Freq_Patterns, Freq, output_file)