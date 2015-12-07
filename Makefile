CXXC = g++
CFLAGS = -Wall -O3 -fPIC

all: libtsvm
  
libtsvm: ssl.o
	$(CXXC) -shared -o libtsvm.so ssl.o
ssl.o: ssl.cpp ssl.h
	$(CXXC) $(CFLAGS) -c ssl.cpp
clean:
	rm -f *~ ssl.o libtsvm.so
