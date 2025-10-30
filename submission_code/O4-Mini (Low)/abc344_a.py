def main():
    S = input().rstrip()
    first = S.find('|')
    last = S.rfind('|')
    # Construct the result by excluding the segment between first and last (inclusive)
    result = S[:first] + S[last+1:]
    print(result)

if __name__ == "__main__":
    main()