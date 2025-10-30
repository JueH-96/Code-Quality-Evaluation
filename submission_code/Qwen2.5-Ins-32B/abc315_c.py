import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    fs = list(zip(map(int, data[1::2]), map(int, data[2::2])))
    
    flavor_dict = defaultdict(list)
    for f, s in fs:
        flavor_dict[f].append(s)
    
    max_satisfaction = 0
    max_s = 0
    second_max_s = 0
    for flavor, scores in flavor_dict.items():
        scores.sort(reverse=True)
        if len(scores) > 1:
            max_satisfaction = max(max_satisfaction, scores[0] + scores[1] // 2)
        if scores[0] > max_s:
            second_max_s = max_s
            max_s = scores[0]
        elif scores[0] > second_max_s:
            second_max_s = scores[0]
    
    max_satisfaction = max(max_satisfaction, max_s + second_max_s)
    
    print(max_satisfaction)

if __name__ == "__main__":
    main()