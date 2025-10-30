import sys

def main():
    n = int(sys.stdin.readline())
    p_line = list(map(int, sys.stdin.readline().split()))
    q_line = list(map(int, sys.stdin.readline().split()))
    
    q_map = [0] * (n + 1)  # Indexes 1..n will be used
    for x in range(1, n + 1):
        q_val = q_line[x - 1]
        q_map[q_val] = x  # q_map[bib_number] = person's original number
    
    result = []
    for i in range(1, n + 1):
        person_wearing_i = q_map[i]
        target_person = p_line[person_wearing_i - 1]
        s_i = q_line[target_person - 1]
        result.append(str(s_i))
    
    print(' '.join(result))

if __name__ == "__main__":
    main()