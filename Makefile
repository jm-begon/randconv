SRCDIR=randconv
DOCDIR=doc

all: clean in install test clean

clean:
	python setup.py clean
	find . -name .DS_Store -delete
	find . -name *.pyc -delete
	rm -rf build

in: inplace

inplace:
	python setup.py build_ext --inplace

doc: inplace
	$(MAKE) -C "$(DOCDIR)" html

cython:
	find "$(SRCDIR)" -name "*.pyx" -exec cython {} \;

test:
	nosetests "$(SRCDIR)" -sv --with-coverage

install:inplace
	python setup.py install