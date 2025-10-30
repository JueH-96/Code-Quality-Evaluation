import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    t = int(data[1])
    events = []
    index = 2
    for i in range(t):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        events.append((a, b))
    
    scores = [0] * (n + 1)
    freq = {}
    freq[0] = n
    distinct_count = 1
    
    out_lines = []
    
    for a, b in events:
        old = scores[a]
        new = old + b
        
        count_old = freq[old]
        if count_old == 1:
            del freq[old]
            distinct_count -= 1
        else:
            freq[old] = count_old - 1
        
        if new in freq:
            freq[new] += 1
        else:
            freq[new] = 1
            distinct_count += 1
            
        scores[a] = new
        out_lines.append(str(distinct_count))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()