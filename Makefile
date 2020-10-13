analysis:
	mkdir -p figures
	mkdir -p stats
	./dbc_rerp_analysis.py

ratings:
	mkdir -p figures
	./dbc_ratings_density.py

clean:
	rm -rf figures
	rm -rf stats
