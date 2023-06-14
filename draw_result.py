import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1, 11, 1)
y=np.array([14.617882926144134,15.805228162338555,16.5830101087571,17.474763212790236,17.48843212341115,
   18.55661253442286,18.56033083409248,18.771307942021405,18.568167034997987,18.593729943315697])
y_=y/19*100
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y,'r',marker='v');
ax1.legend(loc=1)
ax1.set_ylabel('Fp');
ax1.set_xlabel('p');
ax2 = ax1.twinx() 
ax2.plot(x, y_, 'r',marker='v')
ax2.legend(loc=2)
ax2.set_xlim([0.5,10.5]);
ax2.set_ylabel('Accuracy(%)');
ax2.set_xlabel('p');
fig.savefig("result_8.png")
plt.show()

x = np.arange(1, 10, 1)
y=np.array([8.216762096456371,9.46889724562776,10.220378340618783,10.631847440104444,10.812714448698967,
            10.8865571094276,10.93978929290213,10.939841111491877,10.9729959784368])
y_=y/11*100
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y,'r',marker='v');
ax1.legend(loc=1)
ax1.set_ylabel('Fp');
ax1.set_xlabel('p');
ax2 = ax1.twinx() 
ax2.plot(x, y_, 'r',marker='v')
ax2.legend(loc=2)
ax2.set_xlim([0.5,10.5]);
ax2.set_ylabel('Accuracy(%)');
ax2.set_xlabel('p');
fig.savefig("result_7.png")
plt.show()
