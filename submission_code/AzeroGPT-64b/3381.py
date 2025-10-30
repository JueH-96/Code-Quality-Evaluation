class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        if nums[0] | k == k:
            return 1
        n=str(bin(k))[2:]
        ans= inf
        for i in range(1,len(nums)):
            nums[i]|=nums[i-1]
        for j in range(len(nums)):
            for r in range(j,len(nums)):
                x=str(bin(nums[r]))
                y=x[2:] if j==0 else str(bin(nums[j]-1))[2:]
                y="0"*(len(n)-len(y))+y
                x="0"*(len(n)-len(x))+x[2:]
                cnt=0
                for s,g in zip(n,y):
                    if int(s) and (int(s)!=int(g) or int(s)!=int(x[cnt])):
                        break
                    cnt+=1
                else:
                    ans=min(ans,r-j+1)
                    
        return ans if ans!=inf else -1