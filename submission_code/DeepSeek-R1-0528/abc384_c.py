def main():
    scores = list(map(int, input().split()))
    base = "ABCDE"
    participants = []
    
    for bitmask in range(1, 32):
        name_list = []
        total_score = 0
        for i in range(5):
            if bitmask & (1 << i):
                name_list.append(base[i])
                total_score += scores[i]
        name_str = ''.join(name_list)
        participants.append((name_str, total_score))
    
    participants.sort(key=lambda x: (-x[1], x[0]))
    
    for name, _ in participants:
        print(name)

if __name__ == '__main__':
    main()