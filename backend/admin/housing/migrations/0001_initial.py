# Generated by Django 3.2.7 on 2021-10-01 09:17

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Housing',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('longitude', models.FloatField()),
                ('latitude', models.FloatField()),
                ('housing_median_age', models.FloatField()),
                ('total_rooms', models.FloatField()),
                ('total_bedrooms', models.FloatField()),
                ('population', models.FloatField()),
                ('households', models.FloatField()),
                ('median_income', models.FloatField()),
                ('median_house_value', models.FloatField()),
                ('ocean_proximity', models.FloatField()),
            ],
            options={
                'db_table': 'housing',
            },
        ),
    ]
