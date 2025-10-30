def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    def is_polish(seq):
        if not seq:
            return False
        if len(seq) == 1 and seq[0] == 0:
            return True
        
        v1 = seq[0]
        if v1 > len(seq) - 1:
            return False
        
        sub_sequences = []
        start = 1
        for _ in range(v1):
            
            found = False
            for length in range(1, len(seq) - start + 1):
                sub_seq = seq[start:start+length]
                if is_polish(sub_seq):
                    sub_sequences.append(sub_seq)
                    start += length
                    found = True
                    break
            if not found:
                return False
        
        if start != len(seq):
            return False
        
        return True

    def generate_sequences(length):
        if length == 0:
            return [[]]
        
        sequences = []
        for first_val in range(n):
            for sub_sequence in generate_sequences(length - 1):
                sequences.append([first_val] + sub_sequence)
        return sequences

    count = 0
    
    
    def is_lexicographically_smaller_or_equal(seq1, seq2):
        len1 = len(seq1)
        len2 = len(seq2)
        
        if len1 < len2 and seq1 == seq2[:len1]:
            return True
        
        for i in range(min(len1, len2)):
            if seq1[i] < seq2[i]:
                return True
            elif seq1[i] > seq2[i]:
                return False
        
        if len1 <= len2:
            return True
        else:
            return False

    
    
    all_sequences = []
    def generate_sequences_recursive(current_sequence, length):
        if len(current_sequence) == length:
            all_sequences.append(current_sequence.copy())
            return

        for i in range(n):
            current_sequence.append(i)
            generate_sequences_recursive(current_sequence, length)
            current_sequence.pop()

    generate_sequences_recursive([], n)
    
    
    for seq in all_sequences:
        if is_polish(seq) and is_lexicographically_smaller_or_equal(seq, a):
            count = (count + 1) % 998244353
    
    print(count)

solve()