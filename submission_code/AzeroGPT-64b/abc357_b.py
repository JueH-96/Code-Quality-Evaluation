S = list(input())
S = [x.lower() if x.isupper() else x.upper() for x in S if S.count(x.lower())<=S.count(x.upper())]
S = [x.lower() if x.islower() else x.upper() for x in S if S.count(x.lower())>S.count(x.upper())]
print(''.join(S))