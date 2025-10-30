def main():
    data = input().split()
    a, b, c, d, e = map(int, data)
    scores = [a, b, c, d, e]
    problems = "ABCDE"
    participants = []
    
    for mask in range(1, 32):
        name = ""
        total_score = 0
        for i in range(5):
            if mask & (1 << i):
                name += problems[i]
                total_score += scores[i]
        participants.append((total_score, name))
    
    participants.sort(key=lambda x: (-x[0], x[1]))
    
    for _, name in participants:
        print(name)

if __name__ == "__main__":
    main()