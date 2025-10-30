def main():
    s = input().split()
    ab, ac, bc = s[0], s[1], s[2]
    
    oldest = None
    if ab == '>' and ac == '>':
        oldest = 'A'
    elif ab == '<' and bc == '>':
        oldest = 'B'
    elif ac == '<' and bc == '<':
        oldest = 'C'
    
    youngest = None
    if ab == '<' and ac == '<':
        youngest = 'A'
    elif ab == '>' and bc == '<':
        youngest = 'B'
    elif ac == '>' and bc == '>':
        youngest = 'C'
    
    brothers = ['A', 'B', 'C']
    brothers.remove(oldest)
    brothers.remove(youngest)
    print(brothers[0])

if __name__ == "__main__":
    main()