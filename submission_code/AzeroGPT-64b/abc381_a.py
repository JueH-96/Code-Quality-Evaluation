N = int(input())
S = list(input())
ans = ""
lt = N//2
if N%2==1:
    if S[lt]=="/":
        flag = 0
        for i in range(lt):
            if S[i]!=="1":
                flag = 1
        for i in range(lt+1,N):
            if S[i]!=="2":
                flag = 1
        if flag==0:
            ans = "Yes"
        else:
            ans = "No"
    else:
        ans = "No"
else:
    ans = "No"
print(ans)