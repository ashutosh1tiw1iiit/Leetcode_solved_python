class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        if len(nums) == 0 or len(nums) == 1:
            return []
        
        nums.sort()
        lst = []
        for i in range(0, len(nums)):
            
            start = i+1
            end = len(nums) -1
            
            if (i> 0  and nums[i] == nums[i-1]):
                continue
            
            while(start < end):
                
                if (end < len(nums) -1 and nums[end] == nums[end+1]):
                    end -=1
                    continue
                sum3  = nums[i] + nums[start] + nums[end]
                if ( sum3 == 0):
                    lst.append([nums[i], nums[start], nums[end]])
                    start +=1
                    end -=1
                elif sum3 < 0:
                    start +=1
                else:
                    end -=1
        return lst
      
      
      
      class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        
        
        ans = 0
        la = len(A)
        lb = len(B)
        lc = len(C)
        ld = len(D)
        m = {}
        
        for i in range(0, la):
            x = A[i]
            for j in range(0, lb):
                y = B[j]
                if x+y not in m:
                    m[x+y] = 0
                m[x+y] +=1
            
        for i in range(0, lc):
            x = C[i]
            for j in range(0, ld):
                y = D[j]
                
                target = -(x+y)
                
                if target in m:
                    
                    ans += m[target]
        return ans
      
      
      class Solution:
    def addBinary(self, a: str, b: str) -> str:
        lst = []
        i = len(a) - 1
        j = len(b) - 1
        carry = 0
        while(i >=0 or j>=0):
            
            current_sum = carry
            
            if i >=0:
                current_sum += int(a[i])
                i -=1
            if  j>=0:
                current_sum +=int(b[j])
                j-=1
                
            lst.insert(0, current_sum % 2)
            carry = current_sum // 2
        
        if carry > 0:
            lst.insert(0, carry)
        
        return ''.join(map(str, lst))
      
      class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        
        i = len(num1) - 1
        j = len(num2) - 1
        lst = []
        carry = 0
        sum1 = 0
        while( i >=0 or j >=0):
            sum1 = carry
            if  i >=0:
                sum1 += int(num1[i])
                i -=1
            if  j >=0:
                sum1 += int(num2[j])
                j -=1
                
            lst.insert(0, sum1 % 10)
            carry = sum1 // 10
            
        if carry !=0:
            lst.insert(0, 1)
            
        return ''.join(map(str, lst))
      
      
      
      # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        resultNode = ListNode(-1)
        head = resultNode
        p = l1
        q = l2
        sum1 = 0
        carry = 0
        while p is not None or q is not None:
            x = p.val if p is not None else 0
            y = q.val if q is not None else 0
            
            sum1 = carry + x + y
            
            head.next = ListNode(sum1 %10)
            carry = sum1//10
            head = head.next
            
            if p is not None:
                p = p.next
            if q is not None:
                q = q.next
                
        if carry > 0:
            head.next = ListNode(1)
            
        return resultNode.next
