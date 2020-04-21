build:
	go build ./...

run-tests:
	cd tests && go test -o ../tests.test -c -covermode=atomic -coverprofile=c.out -coverpkg=../...
	./tests.test -test.v=true -test.coverprofile=c.out
	sed -i -e '\:^github.com/sudachen/go-dl/:d' c.out
	sed -i -e '\:^github.com/sudachen/go-iokit/:d' c.out
	sed -i -e '\:^github.com/sudachen/go-zorros/:d' c.out
	cp c.out gocov.txt
	sed -i -e 's:github.com/sudachen/go-ml/::g' c.out

run-cover:
	go tool cover -html=gocov.txt

run-cover-tests: run-tests run-cover

