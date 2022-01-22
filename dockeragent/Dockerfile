FROM python:3.6

ADD /examples/dockeragent  /agent
ADD . /pommerman
RUN cd /pommerman && pip --no-cache-dir install -r requirements.txt

# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /pommerman
ENTRYPOINT ["python"]
CMD ["run.py"]