analysis:
	mkdir -p figures
	mkdir -p stats
	python dbc_rerp_analysis.py

ratings:
	mkdir -p figures
	python dbc_ratings_density.py

clean:
	rm -rf figures
	rm -rf stats
