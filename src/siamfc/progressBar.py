import sys
import time

class SimpleProgressBar():
    def __init__(self, width=50,total=100,title=''):
        self.last_x = -1
        self.width = width
        self.total = total
        self.title = title
        
    def update(self, x):
        #assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        if x>self.total: x=self.total
        
        pointer = int(self.width * (x/self.total))
        percent = round(x*100/self.total,2)
        #sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        #sys.stdout.write( '\r%d%% [%s]' % (int(x), '=' * pointer + '>' + '.' * (self.width - pointer)))
        #sys.stdout.write( '\r%d%% %d/%d [%s]' % (int(x), x, self.total, '=' * pointer + '>' + '.' * (self.width - pointer)))
        
        sys.stdout.write( '\r%s:%d/%d %.2f%% [%s]' % (self.title, x, self.total, percent, '=' * pointer + '>' + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if percent >= 100: 
            print('')
            
    '''
    def update(self, x):
        #assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        #sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        #sys.stdout.write( '\r%d%% [%s]' % (int(x), '=' * pointer + '>' + '.' * (self.width - pointer)))
        #sys.stdout.write( '\r%d%% %d/%d [%s]' % (int(x), x, self.total, '=' * pointer + '>' + '.' * (self.width - pointer)))
        sys.stdout.write( '\r%d/%d %d%% [%s]' % (x, self.total, round(x*100/self.total,2), '=' * pointer + '>' + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100: 
            print('')
    '''
            
def main():
    # An example of usage...
    pb = SimpleProgressBar(total=200)
    for i in range(201):
        pb.update(i)
        time.sleep(0.05)
    
if __name__=='__main__':
    main()
    