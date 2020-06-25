from s3.reader import S3Reader


# standalone process to load CSV file data from AWS S3
def main():
    r = S3Reader(
        bucket_name="",
        key=""
    )
    df = r.execute()


if __name__ == '__main__':
    main()
