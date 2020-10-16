analysis:
	mkdir -p figures
	mkdir -p stats
	python3 dbc_rerp_analysis.py

ratings:
	mkdir -p figures
	python3 dbc_ratings_density.py

clean:
	rm -rf figures
	rm -rf stats
