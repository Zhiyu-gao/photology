from django.db import models
# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class A91(models.Model):
    c = models.FloatField(blank=True, null=True)
    b = models.FloatField(blank=True, null=True)
    d = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'a91'


class C21(models.Model):
    wavelength = models.FloatField(blank=True, null=True)
    theta = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'c21'


class C22(models.Model):
    wavelength = models.FloatField(blank=True, null=True)
    xc = models.FloatField(blank=True, null=True)
    yc = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'c22'


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Psfmtf(models.Model):
    u0 = models.FloatField(blank=True, null=True)
    v0 = models.FloatField(blank=True, null=True)
    w0 = models.FloatField(blank=True, null=True)
    w040 = models.FloatField(blank=True, null=True)
    w131 = models.FloatField(blank=True, null=True)
    w222 = models.FloatField(blank=True, null=True)
    w220 = models.FloatField(blank=True, null=True)
    w311 = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'psfmtf'


class Rpw(models.Model):
    t1 = models.FloatField(blank=True, null=True)
    n1 = models.FloatField(blank=True, null=True)
    n2 = models.FloatField(blank=True, null=True)
    rl = models.FloatField(db_column='rL', blank=True, null=True)  # Field name made lowercase.
    a = models.JSONField(db_column='A', blank=True, null=True)  # Field name made lowercase.
    b = models.JSONField(db_column='B', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'rpw'


class Rsw(models.Model):
    th4 = models.IntegerField(blank=True, null=True)
    th3 = models.IntegerField(blank=True, null=True)
    lensr = models.IntegerField(db_column='lensR', blank=True, null=True)  # Field name made lowercase.
    th1 = models.IntegerField(blank=True, null=True)
    th2 = models.IntegerField(blank=True, null=True)
    a = models.JSONField(db_column='A', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'rsw'
# Create your models here.
