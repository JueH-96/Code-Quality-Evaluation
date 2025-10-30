import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    
    scores = [0] * N
    freq = {0: N}
    
    output = []
    
    index = 2
    for _ in range(T):
        A_i = int(data[index])
        B_i = int(data[index+1])
        index += 2
        player = A_i - 1
        old_score = scores[player]
        freq[old_score] -= 1
        if freq[old_score] == 0:
            del freq[old_score]
        new_score = old_score + B_i
        if new_score in freq:
            freq[new_score] += 1
        else:
            freq[new_score] = 1
        scores[player] = new_score
        output.append(str(len(freq)))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()